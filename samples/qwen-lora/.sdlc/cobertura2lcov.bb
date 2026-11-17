#!/usr/bin/env bb
;; Convert kcov's Cobertura report to lcov line records (SF/DA/LF/LH) for
;; sdlc.bb's gate. kcov measures lines only, so there are no BRDA records
;; (:coverage-kind :c0). Usage: cobertura2lcov.bb <cobertura.xml> > lcov.info
(require '[clojure.data.xml :as x] '[clojure.string :as str])
(let [doc (x/parse-str (slurp (first *command-line-args*)) :support-dtd false)
      els (fn els [n] (cons n (mapcat els (filter map? (:content n)))))
      all (els doc)
      src (or (some #(when (= :source (:tag %)) (str/trim (str (first (:content %))))) all) "")]
  (doseq [c (filter #(= :class (:tag %)) all)
          :let [f (get-in c [:attrs :filename])
                ls (for [l (els c) :when (= :line (:tag l))]
                     [(get-in l [:attrs :number]) (parse-long (get-in l [:attrs :hits]))])]]
    (println (str "SF:" (if (str/starts-with? f "/") f (str src f))))
    (doseq [[n h] ls] (println (str "DA:" n "," h)))
    (println (str "LF:" (count ls)))
    (println (str "LH:" (count (filter #(pos? (second %)) ls))))
    (println "end_of_record")))
