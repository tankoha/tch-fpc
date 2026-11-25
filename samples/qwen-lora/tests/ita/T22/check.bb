;; ITa driver of T22 (written by the harness owner; agents run it through
;; tests/ita/T22-run.sh, they do not change it).
;; Usage: bb check.bb EXPECTED VERIFY_TSV LEFT_OVER SECONDS T19_SUMMARY T19_REJECTS HOME
;; Prints one line per case: case<TAB>value<TAB>limit<TAB>note.
(require '[cheshire.core :as json] '[clojure.java.io :as io] '[clojure.string :as str])
(defn emit [c v lim note] (println (str c "\t" v "\t" lim "\t" note)))
(defn text [f] (if (.exists (io/file f)) (slurp f) ""))
(defn lines [f] (remove str/blank? (str/split-lines (text f))))
(let [[exp-f ver-f left secs t19-f rej-f home] *command-line-args*
      exp (into {} (map #(str/split % #"\t")) (lines exp-f))
      rows (for [l (lines ver-f) :let [c (str/split l #"\t" -1)]]
             {:name (str/replace (.getName (io/file (first c))) #"\.pas$" "")
              :class (nth c 2 "") :outcome (nth c 3 "") :detail (nth c 4 "")})
      got (into {} (map (juxt :name :outcome)) rows)
      bad1 (+ (count (remove (fn [[n o]] (= o (got n))) exp))
              (count (remove #(= "compiled-run" (:class %)) rows))
              (Math/abs (- (count rows) (count exp))))
      bad2 (count (filter #(or (str/includes? (:detail %) home) (str/includes? (:detail %) "qlora-verify")) rows))
      ng19 (count (filter #(= "NG" (second (str/split % #"\t"))) (lines t19-f)))
      ok19 (count (filter #(= "OK" (second (str/split % #"\t"))) (lines t19-f)))
      rej (first (filter #(= "x-oxygene" (:id %)) (map #(json/parse-string % true) (lines rej-f))))]
  (emit "ITa-T22-001" bad1 0 (format "%d Oxygene files: outcome as planted, class compiled-run (count of mismatches)" (count exp)))
  (emit "ITa-T22-002" bad2 0 "details free of the home directory and the work directory (count of lines)")
  (emit "ITa-T22-003" left 0 "Oxygene work directories left under ~/.cache/qlora-verify by the run (count)")
  (emit "ITa-T22-004" (format "%.2f" (/ (double (parse-long secs)) (max 1 (count rows)))) 5 "seconds per Oxygene file (build and run)")
  (emit "ITa-T22-005" (if (= ok19 6) ng19 (max 1 ng19)) 0 "the T19 ITa with this src/: NG cases (6 OK expected)")
  (emit "ITa-T22-006" (if (= (:reason rej) "verify: wrong-output") 0 1) 0
        (str "T19's x-oxygene compiled, run and rejected: reason " (pr-str (:reason rej)) " (1 if not verify: wrong-output)")))
