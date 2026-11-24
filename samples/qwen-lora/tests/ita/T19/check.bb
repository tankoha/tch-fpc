;; ITa driver of T19 (written by the harness owner; agents run it through
;; tests/ita/T19-run.sh, they do not change it). Compares the output of
;; qlora_dataset (--valid-percent 25, seed 1) with the planted outcomes
;; and with references computed here, independently of the code under test.
;; Usage: bb check.bb RAW EXPECTED.tsv OUT1 OUT2
;; Prints one line per case: case<TAB>value<TAB>limit<TAB>note.
(require '[cheshire.core :as json] '[clojure.java.io :as io] '[clojure.string :as str])
(import '[java.security MessageDigest])
(defn md5 [s] (apply str (map #(format "%02x" %) (.digest (MessageDigest/getInstance "MD5") (.getBytes s "UTF-8")))))
(defn lines [f] (if (.exists (io/file f)) (remove str/blank? (str/split-lines (slurp f))) []))
(defn jl [f] (map #(json/parse-string % true) (lines f)))
(defn emit [c v lim note] (println (str c "\t" v "\t" lim "\t" note)))
(let [[raw expf o1 o2] *command-line-args*
      o1 (str o1 "/") o2 (str o2 "/")
      raw-lines (vec (lines raw))
      expected (mapv #(str/split % #"\t") (lines expf))
      rejects (into {} (map (juxt :line identity)) (jl (str o1 "rejects.jsonl")))
      reason-key (fn [r] (first (str/split (:reason r) #":")))
      ;; 001: every raw line's outcome
      got (vec (for [i (range (count raw-lines))]
                 (if-let [r (get rejects (inc i))] (reason-key r) "accepted")))
      bad1 (count (filter false? (map (fn [[_ w] g] (= w g)) expected got)))
      acc-idx (keep-indexed (fn [i g] (when (= g "accepted") i)) got)
      acc (map #(json/parse-string (raw-lines %) true) acc-idx)
      ;; 002: the split recomputed: per dialect × task type, (n·25 + 50) div 100 by md5("1:" + id)
      want-valid (set (mapcat (fn [[_ rs]] (take (quot (+ (* 25 (count rs)) 50) 100)
                                                   (map :id (sort-by #(md5 (str "1:" (:id %))) rs))))
                              (group-by (juxt :dialect :task_type) acc)))
      train (jl (str o1 "train.jsonl")) valid (jl (str o1 "valid.jsonl"))
      bad2 (+ (count (remove want-valid (map :id valid)))
              (count (filter want-valid (map :id train)))
              (Math/abs (- (count want-valid) (count valid))))
      ;; 003: the records: the raw object plus its class, in raw order
      cls (fn [r] (if (= (:dialect r) "delphi") "fpc-approx" "compiled-run"))
      outs (into {} (map (juxt :id identity)) (concat train valid))
      bad3 (+ (Math/abs (- (count acc) (count outs)))
              (count (remove (fn [r] (= (assoc r :class (cls r)) (get outs (:id r)))) acc))
              (if (= (map :id (concat (filter (comp not want-valid :id) acc)))
                     (map :id train)) 0 1))
      ;; 004: the manifest: one line per accepted record, md5 of its output line, its split
      out-lines (into {} (map (fn [l] [(:id (json/parse-string l true)) l]))
                      (concat (lines (str o1 "train.jsonl")) (lines (str o1 "valid.jsonl"))))
      man (jl (str o1 "manifest.jsonl"))
      bad4 (+ (Math/abs (- (count man) (count acc)))
              (count (remove (fn [m] (and (= (:record_md5 m) (md5 (get out-lines (:id m) "")))
                                         (= (:split m) (if (want-valid (:id m)) "valid" "train"))
                                         (= (:class m) (:class (get outs (:id m))))))
                             man)))
      ;; 005: the card's totals and reason counts against rejects.jsonl
      card (slurp (str o1 "card.md"))
      rc (frequencies (map reason-key (vals rejects)))
      bad5 (+ (if (str/includes? card (format "Raw records: %d. Accepted: %d (train %d, valid %d). Rejected: %d."
                                              (count raw-lines) (count acc) (count train) (count valid) (count rejects))) 0 1)
              (count (remove (fn [[k n]] (str/includes? card (str "| " k " | " n " |"))) rc)))
      ;; 006: a second run gives the same bytes
      files ["train.jsonl" "valid.jsonl" "manifest.jsonl" "rejects.jsonl" "card.md"]
      bad6 (count (remove (fn [f] (= (slurp (str o1 f)) (slurp (str o2 f)))) files))]
  (emit "ITa-T19-001" bad1 0 (format "%d raw records: outcome (accepted, or the reject reason up to its colon) against the planted one (count of mismatches)" (count raw-lines)))
  (emit "ITa-T19-002" bad2 0 (format "%d valid of %d accepted: the split recomputed per dialect x task type (count of mismatches)" (count valid) (count acc)))
  (emit "ITa-T19-003" bad3 0 "train and valid records: the raw object plus its class, in raw order (count of mismatches)")
  (emit "ITa-T19-004" bad4 0 "manifest: one line per accepted record with the MD5 of its output line, its split and class (count of mismatches)")
  (emit "ITa-T19-005" bad5 0 "card: totals and reject reasons against rejects.jsonl (count of mismatches)")
  (emit "ITa-T19-006" bad6 0 "a second run from another directory: the five output files byte-identical (count of files that differ)"))
