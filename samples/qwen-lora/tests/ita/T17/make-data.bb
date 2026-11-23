;; Builds tests/ita/T17/train.jsonl (8 records) and valid.jsonl (4 records),
;; the inputs of the T17 ITa (written by the harness owner; both files are
;; committed, so the ITa does not change when the sources do). Each record
;; is a short piece (about 250 bytes, cut at line ends) of this sample's
;; own sources, prompt "Show the next lines of unit U.", class compiled-run.
;; Usage (from the sample's root): bb tests/ita/T17/make-data.bb
(require '[clojure.string :as str] '[cheshire.core :as json])
(defn pieces [text]
  (loop [lines (str/split-lines text) cur [] n 0 out []]
    (if (empty? lines)
      out
      (let [l (first lines) n2 (+ n (count l) 1)]
        (if (and (>= n2 250) (seq cur))
          (recur (rest lines) [] 0 (conj out (str/join "\n" (conj cur l))))
          (recur (rest lines) (conj cur l) n2 out))))))
(defn recs [unit from n prefix]
  (->> (pieces (slurp (str "src/" unit ".pas")))
       (drop from) (take n)
       (map-indexed (fn [i p] {:id (format "%s%d" prefix i) :dialect "objfpc"
                               :prompt (str "Show the next lines of unit " unit ".")
                               :answer (str "```pascal\n" p "\n```") :class "compiled-run"}))))
(defn write [f rs] (spit f (str (str/join "\n" (map json/generate-string rs)) "\n")))
(write "tests/ita/T17/train.jsonl" (recs "LoraOptim" 2 8 "t"))
(write "tests/ita/T17/valid.jsonl" (recs "LoraOptim" 10 4 "v"))
(println "written")
