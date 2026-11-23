;; Builds tests/ita/T15/records.jsonl, the input of the T15 ITa (written by the
;; harness owner; the file is committed, so the ITa does not change when the
;; sources do). Records are pieces of this sample's own Pascal sources:
;; - pieces of about 1500 bytes cut at line ends, prompt "Show part N of
;;   unit U." (every fifth in Japanese), class compiled-run, every seventh
;;   fpc-approx, every eleventh static-only (not admitted by the ITa);
;; - two whole long units (more than 2048 tokens), class compiled-run.
;; Usage (from the sample's root): bb tests/ita/T15/make-records.bb
(require '[clojure.string :as str] '[cheshire.core :as json] '[babashka.fs :as fs])
(def units ["Checkpoint" "DeltaRule" "GgufWriter" "Lora" "LoraOptim" "QwenChat" "Recompute" "TrainData"])
(def dialects ["objfpc" "fpc-delphi" "unleashed" "delphi"])
(defn pieces [text]
  (loop [lines (str/split-lines text) cur [] n 0 out []]
    (if (empty? lines)
      (if (seq cur) (conj out (str/join "\n" cur)) out)
      (let [l (first lines) n2 (+ n (count l) 1)]
        (if (and (>= n2 1500) (seq cur))
          (recur (rest lines) [] 0 (conj out (str/join "\n" (conj cur l))))
          (recur (rest lines) (conj cur l) n2 out))))))
(def recs
  (let [ps (for [u units
                 :let [f (str "src/" u ".pas")]
                 :when (fs/exists? f)
                 [i p] (map-indexed vector (pieces (slurp f)))]
             [u (inc i) p])]
    (vec (map-indexed
           (fn [k [u n p]]
             (let [k1 (inc k)]
               {:id (format "r%03d" k1)
                :dialect (dialects (mod k 4))
                :prompt (if (zero? (mod k1 5))
                          (str "ユニット " u " の第 " n " 部を示してください。")
                          (str "Show part " n " of unit " u "."))
                :answer (str "```pascal\n" p "\n```")
                :class (cond (zero? (mod k1 11)) "static-only"
                             (zero? (mod k1 7)) "fpc-approx"
                             :else "compiled-run")}))
           ps))))
(def longs
  (for [[i u] (map-indexed vector ["QwenTokenizer" "QwenModel"])]
    {:id (format "long%d" (inc i)) :dialect "objfpc"
     :prompt (str "Show unit " u ".")
     :answer (str "```pascal\n" (slurp (str "src/" u ".pas")) "\n```")
     :class "compiled-run"}))
(spit "tests/ita/T15/records.jsonl"
      (str (str/join "\n" (map #(json/generate-string %) (concat recs longs))) "\n"))
(println (count recs) "pieces," (count longs) "long records")
