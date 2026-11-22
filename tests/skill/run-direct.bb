#!/usr/bin/env bb
;; H39 tasks without an agent: ask an Ollama model directly for the whole
;; program, build and run it, send the compiler errors or the output
;; difference back, and repeat. Measures Pascal/tch-fpc ability apart from
;; tool calling (tests/skill/run-h39.sh needs a model that calls tools).
;;
;; Usage: bb tests/skill/run-direct.bb <model> <with|without> [task...]
;;   OUT     result directory (default: build/h39/direct-<date-time>)
;;   LIBTCH  libtch build directory (default: build/libtch)
;;   ROUNDS  answers per task, the first included (default: 6)
;; "with" puts SKILL.md and reference/*.md of the tch-fpc skill into the
;; system prompt; "without" gives only the task and the feedback.
;; Writes <OUT>/<model>/<mode>/<task>/answer-N.pas, feedback-N.txt and
;; appends "model mode task PASS|FAIL rounds seconds eval-tokens" to
;; <OUT>/summary.tsv. Programs run on the CPU (GPUs hidden).
(require '[babashka.fs :as fs]
         '[babashka.http-client :as http]
         '[babashka.process :as p]
         '[cheshire.core :as json]
         '[clojure.string :as str])

(def root (str (fs/canonicalize (fs/path (fs/parent *file*) ".." ".."))))
(def args *command-line-args*)
(when (< (count args) 2)
  (binding [*out* *err*] (println "usage: run-direct.bb <model> <with|without> [task...]"))
  (System/exit 2))
(def model (first args))
(def mode (second args))
(when-not (#{"with" "without"} mode)
  (binding [*out* *err*] (println "mode is with or without")) (System/exit 2))
(def tasks (or (seq (drop 2 args))
               (sort (map fs/file-name (fs/list-dir (fs/path root "tests/skill/tasks"))))))
(def out (or (System/getenv "OUT")
             (str root "/build/h39/direct-"
                  (.format (java.text.SimpleDateFormat. "yyyyMMdd-HHmmss") (java.util.Date.)))))
(def libtch (str (fs/canonicalize (or (System/getenv "LIBTCH") (str root "/build/libtch")))))
(def rounds (parse-long (or (System/getenv "ROUNDS") "6")))
(def skill-dir (str root "/plugin/skills/tch-fpc"))

(def skill-text
  (str (slurp (str skill-dir "/SKILL.md"))
       (apply str (for [f (sort (fs/glob skill-dir "reference/*.md"))]
                    (str "\n\n---- reference/" (fs/file-name f) " ----\n" (slurp (str f)))))))

(def system-prompt
  (str "You write complete Free Pascal 3.2.2 programs that use tch-fpc (Free Pascal bindings "
       "for libtorch; main unit `Tch`, sources in src/ of the checkout). Use {$mode objfpc}{$H+}. "
       "Answer with exactly one fenced code block (```pascal) holding the whole program; "
       "it is compiled and run as it is. When you get compiler errors or a wrong output, "
       "answer with the whole corrected program again."
       (when (= mode "with")
         (str "\n\nDocumentation of tch-fpc (the tch-fpc skill):\n\n" skill-text))))

(defn chat [messages]
  (let [resp (http/post "http://localhost:11434/api/chat"
                        {:body (json/generate-string {:model model :stream false :messages messages})
                         :headers {"Content-Type" "application/json"}
                         :timeout (* 60 60 1000)})
        body (json/parse-string (:body resp) true)]
    {:content (get-in body [:message :content] "")
     :eval-count (or (:eval_count body) 0)}))

(defn last-code-block [text]
  (let [blocks (re-seq #"(?s)```[A-Za-z]*[ \t]*\n(.*?)```" text)]
    (when (seq blocks) (second (last blocks)))))

(defn sh [dir env & cmd]
  (apply p/shell {:dir dir :out :string :err :string :continue true
                  :extra-env (merge {"HIP_VISIBLE_DEVICES" "-1" "ROCR_VISIBLE_DEVICES" "-1"} env)}
         cmd))

(defn trim-lines [s] (str/join "\n" (map str/trimr (str/split-lines (str/trim (or s ""))))))

;; Returns [:pass nil] or [:fail feedback-text].
(defn check [task-dir answer work]
  (let [b (sh root {"TCH_FPC_ROOT" root "TCH_OUT" (str work "/bin") "TCH_UNITS" (str work "/units")}
              (str skill-dir "/scripts/build-example.sh") answer libtch)]
    (if (not= 0 (:exit b))
      [:fail (str "The program does not compile. Compiler messages:\n"
                  (str/join "\n" (take 30 (filter #(re-find #"(?i)error|fatal|warning" %)
                                                  (str/split-lines (:err b))))))]
      (let [exe (str/trim (:out b))
            r (sh root {} "timeout" "120" exe)
            expected (trim-lines (slurp (str task-dir "/expected.txt")))
            got (trim-lines (:out r))]
        (cond
          (not= 0 (:exit r))
          [:fail (str "The program compiles but fails when run (exit " (:exit r) "). Output:\n"
                      (str/join "\n" (take 20 (str/split-lines got)))
                      "\nError output:\n" (str/join "\n" (take-last 10 (str/split-lines (:err r)))))]
          (not= expected got)
          [:fail (str "The program runs but its output differs. Required output:\n" expected
                      "\n\nYour output:\n" (str/join "\n" (take 30 (str/split-lines got))))]
          :else [:pass nil])))))

(defn run-task [t]
  (let [task-dir (str root "/tests/skill/tasks/" t)
        dir (str out "/" model "/" mode "/" t)
        _ (fs/create-dirs dir)
        start (System/currentTimeMillis)]
    (loop [n 1
           messages [{:role "system" :content system-prompt}
                     {:role "user" :content (slurp (str task-dir "/prompt.md"))}]
           tokens 0]
      (let [{:keys [content eval-count]} (chat messages)
            tokens (+ tokens eval-count)
            code (last-code-block content)
            answer (str dir "/answer-" n ".pas")
            _ (spit (str dir "/reply-" n ".md") content)
            [verdict feedback] (if code
                                 (do (spit answer code) (check task-dir answer (str dir "/work")))
                                 [:fail "Your answer has no fenced code block. Answer with the whole program in one ```pascal block."])
            secs (quot (- (System/currentTimeMillis) start) 1000)]
        (when feedback (spit (str dir "/feedback-" n ".txt") feedback))
        (if (or (= verdict :pass) (>= n rounds))
          (let [line (str/join "\t" [model mode t (if (= verdict :pass) "PASS" "FAIL") n secs tokens])]
            (println line)
            (spit (str out "/summary.tsv") (str line "\n") :append true))
          (recur (inc n)
                 (conj messages {:role "assistant" :content content}
                       {:role "user" :content feedback})
                 tokens))))))

(fs/create-dirs out)
(doseq [t tasks] (run-task t))
