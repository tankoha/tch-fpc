;; The generator driver of M3 (DESIGN 8.16; written by the harness owner).
;; Sends Mistral Large 3 (Bedrock Converse through the AWS CLI) one job at a
;; time with data-spec/ as the system text, checks the returned record with
;; qlora_verify, and returns the verifier's message for at most three
;; repairs. A record that compiled only after a repair also gives a `fix`
;; record (the broken source, the verifier's message, the repaired source).
;; Writes, under OUT (outside the repository):
;;   raw.jsonl   every final record, accepted or not, as the generator wrote
;;               it plus "verify" (the verifier's outcome) and the fix records
;;   log.jsonl   one line per request: job, attempt, tokens, latency, outcome
;;   jobs.edn    the job list (made once; a rerun skips finished jobs)
;; Usage (from the sample's root):
;;   bb gen/gen.bb plan N OUT [SEED]   write OUT/jobs.edn: N jobs per dialect
;;   bb gen/gen.bb run OUT VERIFY [MAX] run the jobs not yet in log.jsonl
;;                                      (at most MAX of them)
;;   bb gen/gen.bb stats OUT            counts, yield, tokens
(require '[clojure.string :as str] '[cheshire.core :as json] '[babashka.process :as p]
         '[clojure.java.io :as io] '[clojure.edn :as edn])

(def model "mistral.mistral-large-3-675b-instruct")
(def temperature 0.7)
(def max-repairs 3)
(def dialects ["objfpc" "fpc-delphi" "unleashed" "delphi" "oxygene"])
(def feature-prefix {"objfpc" "objfpc." "fpc-delphi" "fpcdelphi." "unleashed" "unleashed."
                     "delphi" "delphi." "oxygene" "oxygene."})

(defn spec-text []
  (str/join "\n\n" (map #(slurp (str "data-spec/" %)) ["spec.md" "dialects.md" "topics.md"])))

(defn features-of [dialect]
  (vec (distinct (re-seq (re-pattern (str "`(" (str/replace (feature-prefix dialect) "." "\\.") "[a-z0-9-]+)`"))
                         (slurp "data-spec/dialects.md")))))

(defn topics []
  (vec (map second (re-seq #"(?m)^\| `([a-z-]+)` \|" (slurp "data-spec/topics.md")))))

;; task types in proportion 50/15/15/10/10; fix jobs are write jobs whose
;; first answer failed (spec 8), so the plan asks for write instead of fix
(defn task-type [k]
  (let [m (mod k 20)]
    (cond (< m 13) "write" (< m 16) "extend" (< m 18) "convert" :else "unit")))

(defn plan [n out seed]
  (let [ts (topics)
        r (java.util.Random. seed)
        jobs (vec (for [d dialects
                        :let [fs (mapv second (features-of d))]
                        k (range n)]
                    {:id (format "%s-%05d" (str/replace d "-" "") (inc k))
                     :dialect d :task_type (task-type k)
                     :topic (ts (mod (+ k (.nextInt r (count ts))) (count ts)))
                     :features (vec (distinct [(fs (mod k (count fs))) (fs (mod (+ k 5) (count fs)))]))
                     :language (if (< (.nextDouble r) 0.1) "ja" "en")}))]
    (.mkdirs (io/file out))
    (spit (str out "/jobs.edn") (pr-str jobs))
    (println (count jobs) "jobs in" (str out "/jobs.edn"))))

(defn converse [out system messages]
  ;; one Converse call; returns {:text :usage :latency}
  (let [tmp (str out "/tmp")
        _ (.mkdirs (io/file tmp))
        sf (str tmp "/system.json") mf (str tmp "/messages.json")]
    (spit sf (json/generate-string [{:text system}]))
    (spit mf (json/generate-string messages))
    (let [r (p/sh "aws" "bedrock-runtime" "converse" "--model-id" model
                  "--system" (str "file://" sf) "--messages" (str "file://" mf)
                  "--inference-config" (json/generate-string {:temperature temperature :maxTokens 6000})
                  "--output" "json")]
      (when-not (zero? (:exit r))
        (throw (ex-info (str "converse failed: " (str/trim (:err r))) {:exit (:exit r)})))
      (let [o (json/parse-string (:out r) true)]
        {:text (str/join (map :text (get-in o [:output :message :content])))
         :usage (:usage o) :latency (get-in o [:metrics :latencyMs])}))))

(defn sections [text]
  ;; the four "### name" sections of a reply (spec.md 2), name -> text
  (loop [ls (str/split-lines text) cur nil acc {}]
    (if (empty? ls)
      (into {} (for [[k v] acc] [k (str/join "\n" v)]))
      (let [l (first ls)]
        (if-let [[_ k] (re-find #"^### (prompt|answer|expected_stdout|features)\s*$" l)]
          (recur (rest ls) (keyword k) (assoc acc (keyword k) []))
          (recur (rest ls) cur (if cur (update acc cur conj l) acc)))))))

(defn parse-record [text job attempt]
  ;; the record of DESIGN 8.16 / spec.md 2 from the reply and the job; nil
  ;; when a section is missing or expected_stdout has no ```text fence
  (let [s (sections text)
        exp (some->> (:expected_stdout s) (re-find #"(?s)```[a-z]*\n(.*?)\n?```") second)]
    (when (and (:prompt s) (:answer s) exp)
      {:id (:id job) :dialect (:dialect job) :task_type (:task_type job) :topic (:topic job)
       :features (vec (remove str/blank? (map str/trim (str/split (str (:features s)) #","))))
       :language (:language job) :prompt (str/trim (:prompt s)) :answer (str (str/trim (:answer s)) "\n")
       :expected_stdout (if (= exp "") "" (str exp "\n"))
       :generator {:model model :version "bedrock" :date (str (java.time.LocalDate/now))
                   :prompt_id (:id job) :temperature temperature :repair_attempts attempt}})))

(defn blocks [answer]
  (vec (map second (re-seq #"(?s)```[a-zA-Z]*\n(.*?)```" (or answer "")))))

(defn verify [verify-bin dialect rec]
  ;; writes the record's blocks (units named after themselves) into a fresh
  ;; directory under the home directory and runs qlora_verify on the program
  (let [w (str (System/getProperty "user.home") "/.cache/qlora-gen/" (System/nanoTime))
        bs (blocks (:answer rec))
        unit? #(re-find #"(?i)^\s*unit\s+[A-Za-z_][A-Za-z0-9_.]*\s*;" %)
        prog (first (remove unit? bs))]
    (.mkdirs (io/file w))
    (try
      (if (nil? prog)
        {:outcome "no-program" :detail ""}
        (do (doseq [u (filter unit? bs)]
              (spit (str w "/" (second (re-find #"(?i)^\s*unit\s+([A-Za-z_][A-Za-z0-9_.]*)" u)) ".pas") u))
            (spit (str w "/main.pas") prog)
            (spit (str w "/main.expected") (or (:expected_stdout rec) ""))
            (let [line (str/trim (:out (p/sh {:dir w} verify-bin "--dialect" dialect "main.pas")))
                  c (str/split line #"\t" -1)]
              {:outcome (nth c 3 "") :detail (nth c 4 "") :program prog})))
      (finally (p/sh "rm" "-rf" w)))))

(defn job-text [job]
  (str "Job: " (json/generate-string (select-keys job [:id :dialect :task_type :topic :features :language]))))

(defn repair-text [v]
  (case (:outcome v)
    "compile-error" (str "The verifier could not compile the program. First error: " (:detail v)
                         "\nReturn the whole corrected reply.")
    "wrong-output" "The program ran, but its output differs from expected_stdout. Make the program and expected_stdout agree exactly (every line break included). Return the whole corrected reply."
    "run-error" (str "The program stopped with exit code " (:detail v) ". It must end normally with code 0. Return the whole corrected reply.")
    "timeout" "The program did not finish within the time limit. Return the whole corrected reply."
    "no-sections" "The reply must have exactly the four sections ### prompt, ### answer, ### expected_stdout (with the output inside a ```text fence) and ### features. Return the whole reply again in that format."
    (str "The reply was not accepted (" (:outcome v) " " (:detail v) "). Follow the specification exactly and return the whole corrected reply.")))

(defn run-job [out verify-bin system job logw raww]
  (loop [messages [{:role "user" :content [{:text (job-text job)}]}]
         attempt 0 first-fail nil]
    (let [r (converse out system messages)
          rec (parse-record (:text r) job attempt)
          v (if rec (verify verify-bin (:dialect job) rec) {:outcome "no-sections" :detail ""})
          ok (= "accepted" (:outcome v))]
      (.write logw (str (json/generate-string {:job (:id job) :attempt attempt :usage (:usage r)
                                               :latency (:latency r) :outcome (:outcome v)}) "\n"))
      (.flush logw)
      (let [first-fail (or first-fail (when (= "compile-error" (:outcome v)) {:v v :rec rec}))]
        (if (or ok (>= attempt max-repairs))
          (do (.write raww (str (json/generate-string (assoc (or rec {:id (:id job) :raw (:text r)})
                                                             :verify (:outcome v) :job job)) "\n"))
              (when (and ok first-fail (contains? #{"write" "extend"} (:task_type job)))
                (let [broken (get-in first-fail [:v :program])
                      msg (get-in first-fail [:v :detail])]
                  (.write raww (str (json/generate-string
                                     (assoc rec :id (str (:id job) "-fix") :task_type "fix"
                                            :prompt (str "This program does not compile:\n\n```pascal\n" broken "```\n\n"
                                                         "The compiler says: " msg "\nFix it.")
                                            :broken_source broken :compiler_message msg
                                            :verify "accepted" :job job)) "\n"))))
              (.flush raww)
              (:outcome v))
          (recur (conj messages {:role "assistant" :content [{:text (:text r)}]}
                       {:role "user" :content [{:text (repair-text v)}]})
                 (inc attempt) first-fail))))))

(defn done-jobs [out]
  (let [f (io/file (str out "/raw.jsonl"))]
    (if (.exists f)
      (set (for [l (str/split-lines (slurp f)) :when (not (str/blank? l))
                 :let [o (json/parse-string l true)]]
             (get-in o [:job :id])))
      #{})))

(defn run [out verify-bin mx]
  (let [jobs (edn/read-string (slurp (str out "/jobs.edn")))
        done (done-jobs out)
        todo (take (or mx Long/MAX_VALUE) (remove #(done (:id %)) jobs))
        system (spec-text)]
    (with-open [logw (io/writer (str out "/log.jsonl") :append true)
                raww (io/writer (str out "/raw.jsonl") :append true)]
      (doseq [job todo]
        (let [o (try (run-job out verify-bin system job logw raww)
                     (catch clojure.lang.ExceptionInfo e
                       (binding [*out* *err*] (println "stopped:" (ex-message e)))
                       (System/exit 4)))]
          (println (:id job) o))))))

(defn stats [out]
  (let [ls #(for [l (str/split-lines (slurp (str out "/" %))) :when (not (str/blank? l))] (json/parse-string l true))
        log (ls "log.jsonl") raw (ls "raw.jsonl")
        main (remove #(str/ends-with? (str (:id %)) "-fix") raw)
        by (group-by #(get-in % [:job :dialect]) main)]
    (doseq [[d rs] (sort by)]
      (println (format "%-10s jobs %3d accepted %3d (%.0f %%) fix records %d" d (count rs)
                       (count (filter #(= "accepted" (:verify %)) rs))
                       (* 100.0 (/ (count (filter #(= "accepted" (:verify %)) rs)) (max 1 (count rs))))
                       (count (filter #(and (= d (get-in % [:job :dialect])) (str/ends-with? (str (:id %)) "-fix")) raw)))))
    (println "outcomes" (frequencies (map :verify main)))
    (println "requests" (count log) "input tokens" (reduce + (map #(get-in % [:usage :inputTokens] 0) log))
             "output tokens" (reduce + (map #(get-in % [:usage :outputTokens] 0) log)))
    (println "repairs per job" (frequencies (vals (frequencies (map :job log)))))))

(let [[cmd a b c] *command-line-args*]
  (case cmd
    "plan" (plan (parse-long a) b (parse-long (or c "1")))
    "run" (run a b (some-> c parse-long))
    "stats" (stats a)))
