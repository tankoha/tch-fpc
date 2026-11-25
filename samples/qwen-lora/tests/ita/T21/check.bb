;; ITa driver of T21 (written by the harness owner; agents run it through
;; tests/ita/T21-run.sh, they do not change it). Compares the runs of
;; qlora_logprobs and qlora_train against the oracle of the code before T21
;; (fp32 only) and the bf16 runs against the fp32 ones.
;; Usage: bb check.bb ORACLE_DIR RUNS_DIR
;; Prints one line per case: case<TAB>value<TAB>limit<TAB>note.
(require '[cheshire.core :as json] '[clojure.java.io :as io] '[clojure.string :as str])
(defn emit [c v lim note] (println (str c "\t" v "\t" lim "\t" note)))
(defn text [f] (if (.exists (io/file f)) (slurp f) ""))
(defn lp [f]  ;; id -> vector of {id logprob}, in order
  (into {} (for [l (str/split-lines (text f)) :when (not (str/blank? l))
                 :let [o (json/parse-string l true)]]
             [(:id o) (vec (:top o))])))
(defn updates [f]  ;; the update lines, columns 1-7
  (vec (for [l (str/split-lines (text f)) :when (and (not (str/blank? l)) (not (str/starts-with? l "ckpt")))]
         (str/join "\t" (take 7 (str/split l #"\t"))))))
(defn loss [line] (parse-double (nth (str/split line #"\t") 3)))
(let [[o r] *command-line-args*
      ref-lp (lp (str o "/lp-fp32.jsonl"))
      ref-tr (str/split-lines (text (str o "/train-fp32.tsv")))
      lp32 (text (str r "/lp32.out"))
      a (lp (str r "/lp16a.out"))
      tr32 (updates (str r "/tr32/train-log.tsv"))
      tr16 (updates (str r "/tr16/train-log.tsv"))
      mix (updates (str r "/mix/train-log.tsv"))
      ;; 001: the default kind reproduces the code before T21
      bad1 (+ (if (= lp32 (text (str o "/lp-fp32.jsonl"))) 0 1)
              (Math/abs (- (count tr32) (count ref-tr)))
              (count (remove true? (map = tr32 ref-tr))))
      ;; 002 / 003: bf16 logprobs against the fp32 oracle
      rows (for [[id t] ref-lp
                 :let [b (get a id []) mb (into {} (map (juxt :id :logprob) b))
                       ds (for [p (take 10 t) :let [y (mb (:id p))]] (if y (Math/abs (- (:logprob p) y)) 1e9))]]
             {:d (if (seq ds) (apply max ds) 1e9)
              :top1 (= (:id (first t)) (:id (first b)))
              :margin (- (:logprob (first t)) (:logprob (second t)))})
      v2 (if (seq rows) (apply max (map :d rows)) 1e9)
      bad3 (+ (count (filter #(and (> (:margin %) 0.5) (not (:top1 %))) rows))
              (Math/abs (- (count a) (count ref-lp))))
      ;; 004: bf16 training against fp32, relative loss difference per update
      rel (fn [x y] (/ (Math/abs (- (loss x) (loss y))) (loss y)))
      v4 (if (and (= (count tr16) 4) (= (count tr32) 4)) (apply max (map rel tr16 tr32)) 1e9)
      ;; 005: fp32 to bf16 across a resume
      bad5 (+ (if (= (count mix) 4) 0 1)
              (count (remove true? (map = (take 2 mix) (take 2 tr32))))
              (count (filter #(> % 0.02) (map rel (drop 2 mix) (drop 2 tr32)))))
      ;; 006: bf16 twice, the same bytes
      bad6 (if (and (not= "" (text (str r "/lp16a.out")))
                    (= (text (str r "/lp16a.out")) (text (str r "/lp16b.out")))) 0 1)]
  (emit "ITa-T21-001" bad1 0 (format "default kind: logprobs (%d prompts) and the 4 update lines byte-identical to the code before T21 (count of differences)" (count ref-lp)))
  (emit "ITa-T21-002" (format "%.4f" v2) 0.25 "--compute bf16: largest |logprob difference| over the fp32 top 10 of each prompt")
  (emit "ITa-T21-003" bad3 0 "--compute bf16: prompts whose top-1 token differs although the fp32 margin exceeds 0.5 (count)")
  (emit "ITa-T21-004" (format "%.2e" v4) 0.02 "--compute bf16 training: largest relative loss difference to fp32 over 4 updates")
  (emit "ITa-T21-005" bad5 0 "fp32 for 2 updates, then --compute bf16 --resume: 4 updates, the first 2 identical to fp32, the rest within 0.02 (count of violations)")
  (emit "ITa-T21-006" bad6 0 "--compute bf16 twice: the logprobs byte-identical (1 if not)"))
