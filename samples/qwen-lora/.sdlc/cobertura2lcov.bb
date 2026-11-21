#!/usr/bin/env bb
;; Convert kcov's Cobertura report to lcov line records (SF/DA/LF/LH) for
;; sdlc.bb's gate. kcov measures lines only, so there are no BRDA records
;; (:coverage-kind :c0). Usage: cobertura2lcov.bb <cobertura.xml> > lcov.info
;;
;; A line whose source text is only a comment start ({, (*, //) or blank is
;; left out: FPC attributes code it generates for a generic specialisation
;; in a unit (for example `specialize TDictionary<...>`) to line 1, the
;; unit's header comment, and that code is never run (measured in T05 with
;; a two-line unit; harness owner). Such a line holds no statement.
(require '[clojure.data.xml :as x] '[clojure.string :as str] '[babashka.fs :as fs])
(let [doc (x/parse-str (slurp (first *command-line-args*)) :support-dtd false)
      els (fn els [n] (cons n (mapcat els (filter map? (:content n)))))
      all (els doc)
      src (or (some #(when (= :source (:tag %)) (str/trim (str (first (:content %))))) all) "")
      text-lines (memoize (fn [path] (if (fs/exists? path) (vec (str/split-lines (slurp path))) [])))
      no-code? (fn [path n]
                 (let [t (str/trim (get (text-lines path) (dec n) ""))]
                   (or (= t "") (str/starts-with? t "{") (str/starts-with? t "(*") (str/starts-with? t "//"))))]
  (doseq [c (filter #(= :class (:tag %)) all)
          :let [f (get-in c [:attrs :filename])
                path (if (str/starts-with? f "/") f (str src f))
                ls (for [l (els c) :when (= :line (:tag l))
                         :let [n (parse-long (get-in l [:attrs :number]))]
                         :when (not (no-code? path n))]
                     [n (parse-long (get-in l [:attrs :hits]))])]]
    (println (str "SF:" path))
    (doseq [[n h] ls] (println (str "DA:" n "," h)))
    (println (str "LF:" (count ls)))
    (println (str "LH:" (count (filter #(pos? (second %)) ls))))
    (println "end_of_record")))
