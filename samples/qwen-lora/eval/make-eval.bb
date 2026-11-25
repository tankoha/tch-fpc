;; Builds the evaluation tasks of DESIGN 8.17 from the bundles in eval/src
;; (written by Claude; held out, never training data). A bundle holds, per
;; task:
;;   === <id> <type>            type: write | fix | convert
;;   --- prompt                  the user's request (the closing sentence
;;                               about the reply is added here)
;;   --- given                   fix / convert: the program the prompt shows
;;   --- ref                     the reference solution
;; Writes eval/refs/<id>.pas (and keeps eval/refs/<id>.expected, the
;; reference's checked output) and eval/tasks.jsonl
;; ({id, dialect, type, prompt, expected_stdout, reference}), followed by
;; the regression tasks of eval/src/regression.tsv (id, check, tool or text,
;; prompt; dialect "regression", as qlora_eval reads them).
;; Usage (from the sample's root):
;;   bb eval/make-eval.bb split     write refs/<id>.pas from the bundles
;;   bb eval/make-eval.bb fill      compile and run each reference without
;;                                  .expected and write its output as
;;                                  refs/<id>.expected.new (to be checked by
;;                                  eye, then renamed to .expected)
;;   bb eval/make-eval.bb check V   qlora_verify V on every reference with its
;;                                  .expected: every line must be accepted
;;   bb eval/make-eval.bb givens V  qlora_verify V on the given programs: a fix
;;                                  task's must not compile, a convert task's must
;;                                  compile and run in its own dialect
;;   bb eval/make-eval.bb build     write tasks.jsonl (every task needs .expected)
(require '[clojure.string :as str] '[cheshire.core :as json] '[babashka.process :as p]
         '[clojure.java.io :as io])
(def dialect-of {"objfpc" "objfpc" "fpcdelphi" "fpc-delphi" "unleashed" "unleashed" "delphi" "delphi" "oxygene" "oxygene"})
(def closing
  {"objfpc" "Use Free Pascal with {$mode objfpc}{$H+}."
   "fpc-delphi" "Use Free Pascal with {$mode delphi}."
   "unleashed" "Use Unleashed Pascal with {$mode unleashed}."
   "delphi" "Use Delphi; the program starts with {$APPTYPE CONSOLE}."
   "oxygene" "Use Oxygene (RemObjects Elements, .NET); the file starts with a namespace."})
(defn parse [f]
  (let [d (dialect-of (str/replace (.getName (io/file f)) #"\.txt$" ""))]
    (for [chunk (rest (str/split (slurp f) #"(?m)^=== "))
          :let [[head & body] (str/split-lines chunk)
                [id typ] (str/split (str/trim head) #"\s+")
                secs (loop [ls body cur nil acc {}]
                       (cond (empty? ls) acc
                             (str/starts-with? (first ls) "--- ") (recur (rest ls) (keyword (subs (first ls) 4)) (assoc acc (keyword (subs (first ls) 4)) []))
                             cur (recur (rest ls) cur (update acc cur conj (first ls)))
                             :else (recur (rest ls) cur acc)))
                txt (fn [k] (when-let [v (secs k)] (str (str/trimr (str/join "\n" v)) "\n")))]]
      {:id id :type typ :dialect d :prompt (str/trim (txt :prompt)) :given (txt :given) :ref (txt :ref)})))
(defn all [] (mapcat parse (sort (map str (filter #(str/ends-with? (str %) ".txt") (file-seq (io/file "eval/src")))))))
(defn prompt-of [t]
  (str (:prompt t)
       (when (:given t) (str "\n\n```pascal\n" (:given t) "```"))
       "\n\n" (closing (:dialect t)) " Reply with the complete program in one fenced code block."))
(def home (System/getProperty "user.home"))
(defn run-ref [t]
  ;; compile and run the reference in a fresh directory under the home
  ;; directory (the Elements flatpak has its own /tmp); returns the output
  (let [w (str home "/.cache/qlora-eval-fill/" (:id t))
        _ (p/sh "rm" "-rf" w)
        _ (.mkdirs (io/file w))
        d (:dialect t)
        src (str w "/" (if (= d "oxygene") "Program" (:id t)) ".pas")
        text (if (= d "delphi")
               (-> (:ref t)
                   (str/replace #"\bSystem\.(SysUtils|Classes|Math|StrUtils|DateUtils|Types|Variants|TypInfo|Rtti)\b" "$1")
                   (str/replace #"\bSystem\.Generics\.(Collections|Defaults)\b" "Generics.$1"))
               (:ref t))
        _ (spit src text)
        ul (str home "/unleashed/fpc/bin/fpc")
        fl ["flatpak" "run" "--command=/app/dotnet/dotnet" "com.remobjects.earth"]
        c (case d
            ("objfpc" "fpc-delphi") (p/sh {:dir w} "fpc" "-vew" "-l-" src)
            "unleashed" (p/sh {:dir w} ul "-vew" "-l-" src)
            "delphi" (p/sh {:dir w} ul "-Mdelphiunicode" "-Manonymousfunctions" "-Mfunctionreferences" "-Minlinevars" "-vew" "-l-" src)
            "oxygene" (do (spit (str w "/Ex.elements") (slurp "eval/Ex.elements"))
                          (apply p/sh {:dir w} (concat fl ["/app/earth/EBuild.dll" "Ex.elements"]))))]
    (if (not= 0 (:exit c))
      (str "COMPILE FAILED\n" (str/join "\n" (filter #(re-find #"(?i)error|fatal|^E: " %) (str/split-lines (str (:out c) (:err c))))) "\n")
      (let [r (if (= d "oxygene")
                (apply p/sh {:dir w} (concat fl ["Bin/Ex.dll"]))
                (p/sh {:dir w} (str w "/" (:id t))))]
        (str (:out r) (when (not= 0 (:exit r)) (str "EXIT " (:exit r) "\n")))))))
(let [[cmd v] *command-line-args* ts (all)]
  (.mkdirs (io/file "eval/refs"))
  (let [dup (for [[k n] (frequencies (map :id ts)) :when (> n 1)] k)]
    (when (seq dup) (throw (ex-info (str "duplicate ids " (vec dup)) {}))))
  (case cmd
    "split" (do (doseq [t ts] (spit (str "eval/refs/" (:id t) ".pas") (:ref t)))
                (println (count ts) "references"))
    "fill" (doseq [t ts :let [e (io/file (str "eval/refs/" (:id t) ".expected"))] :when (not (.exists e))]
             (let [o (run-ref t)]
               (spit (str "eval/refs/" (:id t) ".expected.new") o)
               (println (str "== " (:id t) "\n" o))))
    "check" (let [r (p/sh {:dir "eval"} v "refs")
                  ls (remove str/blank? (str/split-lines (:out r)))
                  bad (remove #(str/includes? % "\taccepted\t") ls)]
              (doseq [l bad] (println l))
              (println (count ls) "verified," (count bad) "not accepted"))
    "givens" (let [w (str home "/.cache/qlora-eval-givens")
                   _ (p/sh "rm" "-rf" w)
                   _ (.mkdirs (io/file w))
                   bad (for [t ts :when (:given t)
                             :let [src (str w "/" (:id t) ".pas")
                                   _ (spit src (:given t))
                                   l (str/trim (:out (p/sh {:dir w} v src)))
                                   o (nth (str/split l #"\t") 3 "")
                                   ok (if (= (:type t) "fix") (= o "compile-error") (#{"no-expected" "accepted"} o))]
                             :when (not ok)]
                         l)]
               (doseq [l bad] (println "BAD" l))
               (println (count (filter :given ts)) "given programs," (count bad) "wrong"))
    "build" (with-open [w (io/writer "eval/tasks.jsonl")]
              (doseq [t ts :let [e (io/file (str "eval/refs/" (:id t) ".expected"))]]
                (when-not (.exists e) (throw (ex-info (str "no expected for " (:id t)) {})))
                (.write w (str (json/generate-string {:id (:id t) :dialect (:dialect t) :type (:type t)
                                                      :prompt (prompt-of t) :expected_stdout (slurp e)
                                                      :reference (:ref t)}) "\n")))
              (let [rs (for [l (str/split-lines (slurp "eval/src/regression.tsv")) :when (not (str/blank? l))
                             :let [[id check arg prompt] (str/split l #"\t")]]
                         (cond-> {:id id :dialect "regression" :type check :prompt prompt :check check}
                           (= check "tool-call") (assoc :tool arg)
                           (= check "contains") (assoc :text arg)))]
                (doseq [x rs] (.write w (str (json/generate-string x) "\n")))
                (println (count ts) "Pascal tasks," (count rs) "regression tasks")))))
