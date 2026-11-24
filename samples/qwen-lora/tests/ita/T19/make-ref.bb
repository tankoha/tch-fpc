;; The tokenizer reference of the T19 ITa (written by the harness owner;
;; the output is committed): the bundled llama-server's /tokenize (special
;; tokens parsed, no BOS, as work/oracle/T05/make-oracle.bb) of every raw
;; record's prompt and answer, one line {"id","prompt","answer"} per
;; parseable record; then the answer of x-tokenizer is altered (its last
;; token dropped), the planted tokenizer mismatch. The same script, without
;; the alteration, makes the reference of a real data set at M3.
;; Usage: bb tests/ita/T19/make-ref.bb PORT RAW OUT [--plant]
(require '[babashka.http-client :as http] '[cheshire.core :as json]
         '[clojure.java.io :as io] '[clojure.string :as str])
(let [[port raw out flag] *command-line-args*
      post (fn [text] (-> (http/post (str "http://127.0.0.1:" port "/tokenize")
                                     {:headers {"Content-Type" "application/json"}
                                      :body (json/generate-string {:content text})})
                          :body (json/parse-string true) :tokens))]
  (with-open [w (io/writer out)]
    (doseq [l (line-seq (io/reader raw))
            :let [r (try (json/parse-string l true) (catch Exception _ nil))]
            :when (and (map? r) (string? (:id r)) (string? (:prompt r)) (string? (:answer r)))]
      (let [a (post (:answer r))
            a (if (and (= flag "--plant") (= (:id r) "x-tokenizer")) (vec (butlast a)) a)]
        (.write w (str (json/generate-string {:id (:id r) :prompt (post (:prompt r)) :answer a}) "\n"))))))
