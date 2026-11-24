;; ITa driver of T20 (written by the harness owner; agents run it through
;; tests/ita/T20-run.sh, they do not change it). Checks qlora_eval's output
;; against references computed here: each Pascal reply's first fenced
;; block verified again with qlora_verify (T06), each regression check
;; recomputed, the report's table rows and adoption lines recomputed from
;; the judged results (counts, flips, the exact two-sided sign test).
;; Usage: bb check.bb TASKS BEFORE AFTER OUT1 OUT2 QLORA_VERIFY WORKDIR [SPEED_RATIO]
;; Prints one line per case: case<TAB>value<TAB>limit<TAB>note.
(require '[cheshire.core :as json] '[clojure.java.io :as io] '[clojure.string :as str]
         '[babashka.process :as p])
(defn lines [f] (if (.exists (io/file f)) (remove str/blank? (str/split-lines (slurp f))) []))
(defn jl [f] (map #(json/parse-string % true) (lines f)))
(defn emit [c v lim note] (println (str c "\t" v "\t" lim "\t" note)))
(def verified #{"objfpc" "fpc-delphi" "unleashed" "oxygene"})
(defn first-block [text]
  (loop [ls (str/split-lines (or text "")) inside false acc []]
    (cond (empty? ls) nil
          (not inside) (recur (rest ls) (str/starts-with? (first ls) "```") [])
          (= (str/trimr (first ls)) "```") (apply str (map #(str % "\n") acc))
          :else (recur (rest ls) true (conj acc (first ls))))))
(let [[tasks-f before-f after-f o1 o2 verify work speed] *command-line-args*
      speed (if speed (parse-double speed) ##NaN)
      tasks (into {} (map (juxt :id identity)) (jl tasks-f))
      replies (concat (map #(assoc % :model "before") (jl before-f))
                      (map #(assoc % :model "after") (jl after-f)))
      ;; the reference judgement of every reply
      judge (fn [i r]
              (let [t (tasks (:task r))]
                (if (= (:dialect t) "regression")
                  (let [ok (if (= (:check t) "tool-call")
                             (boolean (some #(= (:name %) (:tool t)) (:tool_calls r)))
                             (str/includes? (:content r) (:text t)))]
                    [(if ok "check-ok" "check-failed") false ok])
                  (if-let [b (first-block (:content r))]
                    (let [dir (str work "/" i)
                          _ (.mkdirs (io/file dir))
                          _ (spit (str dir "/main.pas") b)
                          _ (spit (str dir "/main.expected") (:expected_stdout t))
                          out (:out (p/sh {:dir dir} verify "--dialect" (:dialect t) "main.pas"))
                          outcome (nth (str/split (str/trim out) #"\t") 3)
                          pit (re-find #"(?i)(^|[^a-z0-9-])expect-error:" b)]
                      [outcome (not (#{"compile-error" "static-ok" "static-reject" "skipped"} outcome))
                       (and (= outcome "accepted") (not pit))])
                    ["no-block" false false]))))
      want (map-indexed (fn [i r] (let [[o c p] (judge i r) t (tasks (:task r))]
                                    {:model (:model r) :task (:task r) :seed (:seed r) :think (:think r)
                                     :dialect (:dialect t) :check (:check t) :outcome o :compiled c :passed p}))
                        replies)
      res (rest (lines (str o1 "/results.tsv")))
      row (fn [w] (str/join "\t" [(:model w) (:task w) (:seed w) (if (:think w) "on" "off") (:dialect w)
                                  (:outcome w) (if (:compiled w) 1 0) (if (:passed w) 1 0)]))
      ;; 001: one result line per reply, in order (model, task, seed, think, dialect)
      key5 (fn [l] (str/join "\t" (take 5 (str/split l #"\t"))))
      bad1 (+ (Math/abs (- (count res) (count want)))
              (count (remove true? (map (fn [l w] (= (key5 l) (key5 (row w)))) res want))))
      ;; 002: outcome, compiled, passed of every reply
      bad2 (count (remove true? (map (fn [l w] (= l (row w))) res want)))
      ;; 003: the report's table rows recomputed
      in-col (fn [w col] (cond (= col "verified") (verified (:dialect w))
                               (= (:dialect w) "regression") (= (:check w) col)
                               :else (= (:dialect w) col)))
      cnt (fn [m think col] (let [ws (filter #(and (= (:model %) m) (= (:think %) think) (in-col % col)) want)]
                              {:n (count ws) :c (count (filter :compiled ws)) :p (count (filter :passed ws))}))
      flips (fn [think col]
              (let [b (filter #(and (= (:model %) "before") (= (:think %) think) (in-col % col)) want)
                    a (into {} (map (juxt (juxt :task :seed) identity))
                            (filter #(and (= (:model %) "after") (= (:think %) think)) want))
                    pairs (keep (fn [x] (when-let [y (a [(:task x) (:seed x)])] [x y])) b)]
                {:b (count (filter (fn [[x y]] (and (not (:passed x)) (:passed y))) pairs))
                 :c (count (filter (fn [[x y]] (and (:passed x) (not (:passed y)))) pairs))}))
      cell (fn [{:keys [n c p]} k] (if (zero? n) "-" (str (if (= k :c) c p) "/" n)))
      dialects (sort (distinct (remove #{"regression"} (map :dialect (filter #(= (:model %) "before") want)))))
      cols (concat dialects ["verified" "tool-call" "contains"])
      table-row (fn [think col]
                  (let [cb (cnt "before" think col) ca (cnt "after" think col) f (flips think col)
                        name (if (= col "delphi") "delphi (approximation)" col)
                        reg (#{"tool-call" "contains"} col)]
                    (str "| " name " | " (if reg "-" (cell cb :c)) " | " (cell cb :p)
                         " | " (if reg "-" (cell ca :c)) " | " (cell ca :p) " | " (:b f) " | " (:c f) " |")))
      report (slurp (str o1 "/report.md"))
      rows (for [think [true false] col cols] (table-row think col))
      bad3 (count (remove #(str/includes? report %) rows))
      ;; 004: the adoption lines recomputed
      choose (fn [n k] (reduce (fn [acc i] (/ (* acc (- n i)) (inc i))) 1N (range k)))
      sign-p (fn [b c] (let [n (+ b c)] (if (zero? n) 1.0
                                            (min 1.0 (* 2.0 (/ (double (reduce + (map #(choose n %) (range (inc (min b c))))))
                                                              (Math/pow 2 n)))))))
      rate (fn [{:keys [n p]}] (if (zero? n) ##NaN (/ (double p) n)))
      above (fn [a b] (and (not (Double/isNaN a)) (not (Double/isNaN b)) (> a b)))
      fmt3 (fn [x] (if (Double/isNaN x) "Nan" (format "%.3f" x)))
      adoption (fn [think]
                 (let [rb (rate (cnt "before" think "verified")) ra (rate (cnt "after" think "verified"))
                       f (flips think "verified") p (sign-p (:b f) (:c f))
                       fall (str/join "" (for [d dialects :when (and (verified d)
                                                                     (above (- (rate (cnt "before" think d)) 0.05)
                                                                            (rate (cnt "after" think d))))]
                                           (str " " d)))
                       tb (:p (cnt "before" think "tool-call")) ta (:p (cnt "after" think "tool-call"))
                       conds [[(format "verified aggregate rises (%s -> %s)" (fmt3 rb) (fmt3 ra)) (above ra rb)]
                              [(format "flips significant (%d fail->pass, %d pass->fail, p = %.4f)" (:b f) (:c f) p)
                               (and (> (:b f) (:c f)) (< p 0.05))]
                              [(str "no verified dialect falls by more than 5 points" fall) (= fall "")]
                              [(format "tool-call check loses nothing (%d -> %d)" tb ta) (>= ta tb)]]
                       speed-line (if (Double/isNaN speed) ["H49 speed at least 0.7 of before: not measured" :nm]
                                      [(format "H49 speed at least 0.7 of before (%.3f)" speed) (>= speed 0.7)])
                       all (conj conds speed-line)
                       failed (some #(false? (second %)) all)]
                   (concat (map (fn [[t ok]] (str "- " t ": " (cond (= ok :nm) "not measured" ok "yes" :else "no"))) all)
                           [(str "- adopt: " (cond failed "no" (= (second speed-line) :nm) "undecided" :else "yes"))])))
      ad-lines (concat (adoption true) (adoption false))
      bad4 (count (remove #(str/includes? report %) ad-lines))
      ;; 005: a second run gives the same bytes
      bad5 (count (remove (fn [f] (= (slurp (str o1 "/" f)) (slurp (str o2 "/" f)))) ["results.tsv" "report.md"]))]
  (emit "ITa-T20-001" bad1 0 (format "%d replies: one result line each, in order (count of mismatches)" (count want)))
  (emit "ITa-T20-002" bad2 0 (format "outcome, compiled, passed against qlora_verify on the first block and the recomputed checks; %d passed (count of mismatches)" (count (filter :passed want))))
  (emit "ITa-T20-003" bad3 0 (format "%d table rows of the report recomputed from the judgements (count missing)" (count rows)))
  (emit "ITa-T20-004" bad4 0 (format "%d adoption lines recomputed, sign test included (count missing)" (count ad-lines)))
  (emit "ITa-T20-005" bad5 0 "a second run from another directory: results.tsv and report.md byte-identical (count of files that differ)"))
