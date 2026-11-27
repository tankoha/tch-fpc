;; The generator driver of M3 (DESIGN 8.16; written by the harness owner).
;; Sends Mistral Large 3 (Bedrock Converse through the AWS CLI) one job at a
;; time with data-spec/ as the system text, checks the returned record with
;; qlora_verify, and returns the verifier's message for at most three
;; repairs. A record that compiled only after a repair also gives a `fix`
;; record (the broken source, the verifier's message, the repaired source).
;; For wrong-output the repair message carries the program's actual output.
;; Writes, under OUT (outside the repository):
;;   raw.jsonl   every final record, accepted or not, as the generator wrote
;;               it plus "verify" (the verifier's outcome) and the fix records
;;   log.jsonl   one line per request: job, attempt, tokens, latency, outcome,
;;               the verifier's detail and the actual output (wrong-output)
;;   jobs.edn    the job list (made once; a rerun skips finished jobs)
;; Usage (from the sample's root):
;;   bb gen/gen.bb plan COUNTS OUT [SEED]
;;                                     write OUT/jobs.edn: COUNTS is N (jobs
;;                                     per dialect) or objfpc=N,oxygene=M,...
;;   bb gen/gen.bb plan-more COUNTS OUT SEED [TYPES]
;;                                     append jobs to OUT/jobs.edn (TYPES:
;;                                     e.g. unit,convert; default the mix)
;;   bb gen/gen.bb run OUT VERIFY [MAX [WORKERS [TARGET]]]
;;                                     run the jobs not yet in raw.jsonl (at
;;                                     most MAX of them, WORKERS at a time;
;;                                     with TARGET only jobs whose task-type
;;                                     quota and topic cap are open, so each
;;                                     dialect ends at TARGET accepted
;;                                     non-fix records in spec 3's mix)
;;   bb gen/gen.bb quota OUT TARGET     quotas, yields, expected shortfalls
;;   bb gen/gen.bb reverify OUT VERIFY verify the rejected records again
;;                                     (no Bedrock): errors, actual output
;;   bb gen/gen.bb stats OUT            counts, yield, tokens
;;   bb gen/gen.bb fewshot OUT:ID...    write data-spec/fewshot.jsonl from
;;                                     accepted records of earlier runs
;; Few-shot: before each job the driver puts the example exchanges of the
;; job's dialect from data-spec/fewshot.jsonl (a job line and the accepted
;; reply), all of them generator output, never Claude's (GEN_FEWSHOT=0: none).
;; Repair hints: for compile errors the generator repeated (repair-hints,
;; Oxygene only) the repair request adds one explaining sentence; the log
;; marks such repairs "hint", the record counts them in
;; generator.repair_hints (GEN_HINTS=0: none).
;; The generator: Mistral Large 3 on Bedrock by default; with
;; GEN_MODEL=ollama:<name> a local Ollama model (http://127.0.0.1:11434,
;; the model's own renderer), thinking off unless GEN_THINK=1 (the
;; thinking text is not kept; the record has the answer only).
(require '[clojure.string :as str] '[cheshire.core :as json] '[babashka.process :as p]
         '[clojure.java.io :as io] '[clojure.edn :as edn] '[babashka.http-client :as http])

(def gen-model (or (System/getenv "GEN_MODEL") "mistral.mistral-large-3-675b-instruct"))
(def ollama-model (when (str/starts-with? gen-model "ollama:") (subs gen-model 7)))
(def think (= "1" (System/getenv "GEN_THINK")))
(def model (or ollama-model gen-model))
(def ollama-url (or (System/getenv "OLLAMA_URL") "http://127.0.0.1:11434"))
(def model-version
  ;; Bedrock: "bedrock"; Ollama: the model's digest
  (delay (if ollama-model
           (let [ms (-> (http/get (str ollama-url "/api/tags")) :body (json/parse-string true) :models)
                 m (first (filter #(= ollama-model (:name %)) ms))]
             (when-not m (throw (ex-info (str "ollama has no model " ollama-model) {})))
             (str "ollama " (:digest m) (when think " think")))
           "bedrock")))
(def temperature 0.7)
(def max-repairs 3)
(declare fewshots)
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

(defn counts [spec]
  ;; "N" (N jobs for every dialect) or "objfpc=N,unleashed=M,..." (dialects
  ;; not named get none)
  (if (re-matches #"\d+" spec)
    (zipmap dialects (repeat (parse-long spec)))
    (into {} (for [kv (str/split spec #",") :let [[d n] (str/split kv #"=")]]
               (do (when-not (some #{d} dialects) (throw (ex-info (str "unknown dialect " d) {})))
                   [d (parse-long n)])))))

(defn plan [spec out seed]
  ;; the jobs are drawn dialect by dialect (the pilots' jobs stay as they
  ;; were) and written round-robin over the dialects, so a partial run
  ;; (MAX) advances all dialects together
  (let [ts (topics)
        r (java.util.Random. seed)
        cs (counts spec)
        jobs (vec (for [d dialects
                        :let [fs (mapv second (features-of d))]
                        k (range (get cs d 0))]
                    {:id (format "%s-%05d" (str/replace d "-" "") (inc k))
                     :dialect d :task_type (task-type k)
                     :topic (ts (mod (+ k (.nextInt r (count ts))) (count ts)))
                     :features (vec (distinct [(fs (mod k (count fs))) (fs (mod (+ k 5) (count fs)))]))
                     :language (if (< (.nextDouble r) 0.1) "ja" "en")}))
        jobs (vec (sort-by (juxt #(parse-long (re-find #"\d+$" (:id %)))
                                 #(.indexOf dialects (:dialect %)))
                           jobs))]
    (when (.exists (io/file out "jobs.edn"))
      (throw (ex-info (str out "/jobs.edn exists; a new plan needs a new OUT") {})))
    (.mkdirs (io/file out))
    (spit (str out "/jobs.edn") (pr-str jobs))
    (println (count jobs) "jobs in" (str out "/jobs.edn"))))

(defn chat-ollama [system messages]
  ;; one /api/chat call with the Converse-shaped MESSAGES; returns
  ;; {:text :usage :latency} like the Bedrock call
  (let [body {:model ollama-model :stream false :think think
              :messages (into [{:role "system" :content system}]
                              (for [m messages] {:role (:role m) :content (str/join (map :text (:content m)))}))
              :options {:temperature temperature :num_predict (if think 16000 6000)}}
        r (try (http/post (str ollama-url "/api/chat")
                          {:headers {"Content-Type" "application/json"}
                           :body (json/generate-string body) :timeout 3600000 :throw false})
               (catch Exception e (throw (ex-info (str "ollama failed: " (.getName (class e)) " " (ex-message e)) {}))))]
    (when-not (= 200 (:status r))
      (throw (ex-info (str "ollama failed: HTTP " (:status r) " " (:body r)) {})))
    (let [o (json/parse-string (:body r) true)]
      {:text (str (get-in o [:message :content]))
       :usage {:inputTokens (:prompt_eval_count o 0) :outputTokens (:eval_count o 0)
               :totalTokens (+ (:prompt_eval_count o 0) (:eval_count o 0))}
       :latency (quot (:total_duration o 0) 1000000)})))

(declare converse-bedrock)

(defn converse [tmp system messages]
  (if ollama-model
    (chat-ollama system messages)
    (converse-bedrock tmp system messages)))

(defn converse-bedrock [tmp system messages]
  ;; one Converse call (request files in the worker's TMP); returns
  ;; {:text :usage :latency}
  (let [_ (.mkdirs (io/file tmp))
        sf (str tmp "/system.json") mf (str tmp "/messages.json")]
    (spit sf (json/generate-string [{:text system}]))
    (spit mf (json/generate-string messages))
    (let [call #(p/sh "aws" "bedrock-runtime" "converse" "--model-id" model
                      "--system" (str "file://" sf) "--messages" (str "file://" mf)
                      "--inference-config" (json/generate-string {:temperature temperature :maxTokens 6000})
                      "--output" "json")
          ;; throttling (also of the login token with several workers) is
          ;; retried with a growing pause; any other failure stops the run
          r (loop [k 0]
              (let [r (call)]
                (if (and (not (zero? (:exit r))) (< k 6)
                         (re-find #"Rate exceeded|\(429\)|Throttling|TooManyRequests" (:err r)))
                  (do (Thread/sleep (* 1000 (+ (bit-shift-left 2 k) (rand-int 3))))
                      (recur (inc k)))
                  r)))]
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
       :generator {:model model :version @model-version :date (str (java.time.LocalDate/now))
                   :prompt_id (:id job) :temperature temperature :repair_attempts attempt
                   :fewshot (mapv :source (get @fewshots (:dialect job)))}})))

(defn blocks [answer]
  (vec (map second (re-seq #"(?s)```[a-zA-Z]*\n(.*?)```" (or answer "")))))

;; the Oxygene project file of the verifier (K_OXY_PROJECT in
;; src/DialectVerify.pas), used to rebuild a program whose output was wrong
(def oxy-project
  (str/join "\n"
            ["<Project DefaultTargets=\"Build\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\" ToolsVersion=\"4.0\">"
             "  <PropertyGroup>" "    <RootNamespace>Ex</RootNamespace>"
             "    <ProjectGuid>{6b1f1c3e-1d2a-4c55-9a7e-0c1f2e3d4a01}</ProjectGuid>"
             "    <OutputType>Exe</OutputType>" "    <AssemblyName>Ex</AssemblyName>"
             "    <Configuration Condition=\"'$(Configuration)' == ''\">Release</Configuration>"
             "    <TargetFramework>.NETCore</TargetFramework>" "    <Mode>Echoes</Mode>"
             "    <OutputPath>.\\Bin</OutputPath>" "  </PropertyGroup>"
             "  <ItemGroup>" "    <Compile Include=\"Program.pas\" />" "  </ItemGroup>"
             "  <Import Project=\"$(MSBuildExtensionsPath)\\RemObjects Software\\Elements\\RemObjects.Elements.Echoes.targets\" />"
             "</Project>" ""]))

(def actual-max 3000)

(defn actual-output [w dialect prog]
  ;; the stdout of a program the verifier judged wrong-output: the FPC build
  ;; stays in w/work/verify/<n>/main; Oxygene's is deleted, so rebuild it
  (let [run (fn [dir secs args]
              (let [r (apply p/sh {:dir dir :continue true} "timeout" (str secs) args)]
                (when (zero? (:exit r)) (:out r))))
        out (if (= dialect "oxygene")
              (let [d (str w "/oxy")]
                (.mkdirs (io/file d))
                (spit (str d "/Program.pas") prog)
                (spit (str d "/Ex.elements") oxy-project)
                (when (run d 300 ["flatpak" "run" "--command=/app/dotnet/dotnet" "com.remobjects.earth"
                                  "/app/earth/EBuild.dll" "Ex.elements"])
                  (run d 20 ["flatpak" "run" "--command=/app/dotnet/dotnet" "com.remobjects.earth" "Bin/Ex.dll"])))
              (when-let [exe (first (filter #(and (= "main" (.getName %)) (.canExecute %))
                                            (file-seq (io/file w "work/verify"))))]
                (run (.getParent exe) 20 [(.getPath exe)])))]
    (when out
      (if (> (count out) actual-max) (str (subs out 0 actual-max) "\n[... cut]\n") out))))

(def oxy-lock (Object.))

(defn error-context [detail bs prog]
  ;; the lines around the compiler's first error, numbered, the error line
  ;; marked and (FPC) a caret under the column; nil when the place is unknown
  (let [[_ f ln col] (or (re-find #"^([A-Za-z0-9_.]+)\((\d+),(\d+)\)" detail)
                         (re-find #"\[(Program\.pas) \((\d+)\)\]" detail))
        src (cond (nil? f) nil
                  (#{"main.pas" "Program.pas"} f) prog
                  :else (first (filter #(re-find (re-pattern (str "(?i)^\\s*unit\\s+"
                                                                   (java.util.regex.Pattern/quote (str/replace f #"\.pas$" ""))
                                                                   "\\s*;")) %) bs)))]
    (when src
      (let [ls (str/split-lines src) n (parse-long ln)]
        (when (<= 1 n (count ls))
          (str/join "\n"
                    (for [i (range (max 1 (- n 3)) (inc (min (count ls) (+ n 2))))
                          s (cons (format "%4d%s %s" i (if (= i n) " >>" "   ") (nth ls (dec i)))
                                  (when (and col (= i n))
                                    [(str (apply str (repeat (+ 8 (dec (parse-long col))) " ")) "^")]))]
                      s)))))))

(defn verify [verify-bin dialect rec]
  ;; writes the record's blocks (units named after themselves) into a fresh
  ;; directory under the home directory and runs qlora_verify on the program;
  ;; for wrong-output also returns the program's actual output. Oxygene runs
  ;; one at a time (the verifier's build directory is named after the clock)
  (let [w (str (System/getProperty "user.home") "/.cache/qlora-gen/" (System/nanoTime)
               "-" (.getId (Thread/currentThread)))
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
            (let [go (fn []
                       (let [line (str/trim (:out (p/sh {:dir w} verify-bin "--dialect" dialect "main.pas")))
                             c (str/split line #"\t" -1)
                             v {:outcome (nth c 3 "") :detail (nth c 4 "") :program prog}]
                         (case (:outcome v)
                           "wrong-output" (assoc v :actual (actual-output w dialect prog))
                           "compile-error" (assoc v :context (error-context (:detail v) bs prog))
                           v)))]
              (if (= dialect "oxygene") (locking oxy-lock (go)) (go)))))
      (finally (p/sh "rm" "-rf" w)))))

(defn job-text [job]
  (str "Job: " (json/generate-string (select-keys job [:id :dialect :task_type :topic :features :language]))))

(defn error-key [detail]
  ;; the compiler message without file and position, to see a repeated error
  (-> (str detail) (str/replace #"^[A-Za-z0-9_.]+\(\d+,\d+\)\s*" "")
      (str/replace #"\s*\[Program\.pas \(\d+\)\]$" "")))

;; Repair hints (DESIGN 8.16, 2026-11-27): a sentence added to the repair
;; request when the compiler's first error matches, for mistakes the
;; generator repeated and did not fix from the message alone (the counts of
;; run 1). Prose with single constructs, as in dialects.md's "Common
;; mistakes"; checked against the compiler.
(def repair-hints
  {"oxygene"
   [[#"expected, got \"implementation\""
     (str "The file has an `implementation` section but no `interface` line. An Oxygene file uses one of two layouts: "
          "`namespace X;` `interface` `uses ...;` `type ...` (declarations only) `implementation` (the bodies, named "
          "`Type.Method`) `end.`, or no `interface`/`implementation` at all and every method body inside its type. "
          "Add `interface` on its own line after the namespace line, or remove `implementation` and move each body into its type.")]
    [#"\"implementation\" or interface section members"
     (str "After `interface` the types hold declarations only. Move each method body below an `implementation` line "
          "(as `method Type.Name(...)`), or remove the `interface` line and keep the bodies inside the types.")]
    [#"identifier expected, got \"result\""
     (str "`result` is the reserved name of a method's return value: do not declare a variable called `result`; "
          "assign `result := ...` directly or give the variable another name.")]
    [#"cannot find operator to evaluate \"T\" [-+*/<>]"
     "A generic `T` has no arithmetic or comparison operators; use a concrete type such as `Double` or `Integer` for values you compute with."]
    [#"cannot find operator to evaluate \"[^\"]+\" [-+]="
     "In Oxygene `+=` and `-=` are only for events; for numbers and strings write `x := x + y` (`inc(x)` adds 1 to an integer)."]
    [#"Unknown identifier \"base\""
     "Oxygene has no `base`; call the parent with `inherited` (in a constructor: `inherited constructor(args)`)."]
    [#"identifier expected, got \"set\""
     "Oxygene properties have no `get`/`set` blocks: write `property X: Integer read fX write fX;`, or `write SetX` with a method `SetX(value: Integer)`."]]})

(def hints-on (not= "0" (System/getenv "GEN_HINTS")))

(defn repair-hint [dialect v]
  (when (and hints-on (= "compile-error" (:outcome v)))
    (some (fn [[re h]] (when (re-find re (str (:detail v))) h)) (get repair-hints dialect))))

(defn repair-text [dialect v prev]
  (case (:outcome v)
    "compile-error" (str "The verifier could not compile the program. First error: " (:detail v)
                         (when (:context v)
                           (str "\n\nThe source around it (>> marks the line):\n\n```\n" (:context v) "\n```"))
                         (when-let [h (repair-hint dialect v)] (str "\n\nHint: " h))
                         (when (and (= "compile-error" (:outcome prev))
                                    (= (error-key (:detail v)) (error-key (:detail prev))))
                           (str "\n\nThis is the same error as in your previous reply: that change did not fix it."
                                " Do not repeat it. Check the construct against the dialect's section and its"
                                " \"Common mistakes\" in the specification, and rewrite that part another way,"
                                " with simpler constructs if needed."))
                         "\nReturn the whole corrected reply.")
    "wrong-output" (str "The program ran, but its output differs from expected_stdout."
                        (if (:actual v)
                          (str " The program printed exactly this:\n\n```text\n" (:actual v)
                               (when-not (str/ends-with? (:actual v) "\n") "\n[no final line break]\n")
                               "```\n\nDecide which is wrong, the program or expected_stdout, and fix it.")
                          "")
                        " Make the program and expected_stdout agree exactly (every line break included). Return the whole corrected reply.")
    "run-error" (str "The program stopped with exit code " (:detail v) ". It must end normally with code 0. Return the whole corrected reply.")
    "timeout" "The program did not finish within the time limit. Return the whole corrected reply."
    "no-sections" "The reply must have exactly the four sections ### prompt, ### answer, ### expected_stdout (with the output inside a ```text fence) and ### features. Return the whole reply again in that format."
    (str "The reply was not accepted (" (:outcome v) " " (:detail v) "). Follow the specification exactly and return the whole corrected reply.")))

(def write-lock (Object.))

(defn write-line [w o]
  ;; one JSON line; several workers share the writers
  (locking write-lock
    (.write w (str (json/generate-string o) "\n"))
    (.flush w)))

(def fewshot-file "data-spec/fewshot.jsonl")

(defn reply-text [rec]
  ;; a record back in the reply format of spec.md section 2
  (str "### prompt\n" (:prompt rec) "\n### answer\n" (:answer rec)
       "\n### expected_stdout\n```text\n" (:expected_stdout rec) "```\n### features\n"
       (str/join ", " (:features rec)) "\n"))

(defn fewshot [srcs]
  ;; writes data-spec/fewshot.jsonl from accepted generator records, each
  ;; SRC as OUT:ID (e.g. work/gen/pilot3:objfpc-00012); the example's job
  ;; line asks only for the job's features the record really lists, and its
  ;; id is example-N (not a real job's id)
  (with-open [w (io/writer fewshot-file)]
    (doseq [[i src] (map-indexed vector srcs)]
      (let [[out id] (str/split src #":")
            rec (first (for [l (str/split-lines (slurp (str out "/raw.jsonl")))
                             :let [r (json/parse-string l true)] :when (= id (:id r))] r))
            _ (when-not (= "accepted" (:verify rec)) (throw (ex-info (str src ": not an accepted record") {})))
            fs (filter (set (:features rec)) (get-in rec [:job :features]))
            job (assoc (:job rec) :id (format "example-%02d" (inc i))
                       :features (vec (if (seq fs) fs (take 3 (:features rec)))))]
        (.write w (str (json/generate-string {:source src :generator (:generator rec)
                                              :job job :reply (reply-text rec)}) "\n"))))))

(def fewshots
  ;; dialect -> the example exchanges put before the job (empty without the file
  ;; or with GEN_FEWSHOT=0)
  (delay (if (or (= "0" (System/getenv "GEN_FEWSHOT")) (not (.exists (io/file fewshot-file))))
           {}
           (group-by #(get-in % [:job :dialect])
                     (for [l (str/split-lines (slurp fewshot-file)) :when (not (str/blank? l))]
                       (json/parse-string l true))))))

(defn fewshot-messages [dialect]
  (vec (mapcat (fn [e] [{:role "user" :content [{:text (job-text (:job e))}]}
                        {:role "assistant" :content [{:text (:reply e)}]}])
               (get @fewshots dialect))))

(declare quotas wanted? tt-share accepted-cells)

(defn plan-more [spec out seed types]
  ;; append jobs to an existing OUT/jobs.edn (a running `run` does not see
  ;; them until it is started again): COUNTS as in plan, numbered on from
  ;; the dialect's last job; TYPES (e.g. "unit,convert") makes them only of
  ;; those task types, in turn; topics, features and language as in plan
  (let [ts (topics)
        r (java.util.Random. seed)
        old (edn/read-string (slurp (str out "/jobs.edn")))
        last-n (fn [d] (reduce max 0 (for [j old :when (= d (:dialect j))] (parse-long (re-find #"\d+$" (:id j))))))
        tys (when types (str/split types #","))
        _ (doseq [t tys] (when-not (tt-share t) (throw (ex-info (str "unknown task type " t) {}))))
        cs (counts spec)
        new (vec (for [d dialects
                       :let [fs (mapv second (features-of d)) n0 (last-n d)]
                       i (range (get cs d 0))
                       :let [k (+ n0 i)]]
                   {:id (format "%s-%05d" (str/replace d "-" "") (inc k))
                    :dialect d :task_type (if tys (tys (mod i (count tys))) (task-type k))
                    :topic (ts (mod (+ k (.nextInt r (count ts))) (count ts)))
                    :features (vec (distinct [(fs (mod k (count fs))) (fs (mod (+ k 5) (count fs)))]))
                    :language (if (< (.nextDouble r) 0.1) "ja" "en")}))
        new (sort-by (juxt #(parse-long (re-find #"\d+$" (:id %))) #(.indexOf dialects (:dialect %))) new)]
    (spit (str out "/jobs.edn") (pr-str (into (vec old) new)))
    (println (count new) "jobs appended," (+ (count old) (count new)) "in" (str out "/jobs.edn"))))


(defn quota-report [out target]
  ;; per dialect and task type: accepted / quota, the yield so far, the
  ;; queued jobs that may still run, and the expected end (have + queued x
  ;; yield; topic caps not counted, so it is an upper estimate); per dialect
  ;; the topics, fewest and most accepted
  (let [q (quotas target)
        ls (for [l (str/split-lines (slurp (str out "/raw.jsonl"))) :when (not (str/blank? l))] (json/parse-string l true))
        main (remove #(str/ends-with? (str (:id %)) "-fix") ls)
        done (set (map #(get-in % [:job :id]) ls))
        cells (accepted-cells out)
        queue (remove #(done (:id %)) (edn/read-string (slurp (str out "/jobs.edn"))))]
    (println "quotas" (:tt q) "topic cap" (:topic q))
    (doseq [d dialects]
      (println d)
      (doseq [[t n] (sort (:tt q))
              :let [rs (filter #(and (= d (get-in % [:job :dialect])) (= t (get-in % [:job :task_type]))) main)
                    have (get cells [:tt d t] 0)
                    y (if (seq rs) (/ (double (count (filter #(= "accepted" (:verify %)) rs))) (count rs)) 0.5)
                    qd (count (filter #(and (= d (:dialect %)) (= t (:task_type %)) (wanted? q cells %)) queue))
                    exp (+ have (* qd y))]]
        (println (format "  %-8s %4d/%-4d yield %3.0f %%  queued %4d  expected %5.0f%s" t have n (* 100 y) qd exp
                         (if (< exp n) (format "  SHORT ~%d jobs" (long (Math/ceil (/ (- n exp) (max y 0.05))))) ""))))
      (let [tc (sort-by second (for [t (topics)] [t (get cells [:topic d t] 0)]))]
        (println "  topics" (str/join ", " (map (fn [[t c]] (str t " " c)) (take 3 tc))) "..."
                 (str/join ", " (map (fn [[t c]] (str t " " c)) (take-last 2 tc))))))))

(defn run-job [tmp verify-bin system job logw raww]
  (loop [messages (conj (fewshot-messages (:dialect job)) {:role "user" :content [{:text (job-text job)}]})
         attempt 0 first-fail nil prev nil hints 0]
    (let [r (converse tmp system messages)
          rec (parse-record (:text r) job attempt)
          rec (when rec (assoc-in rec [:generator :repair_hints] hints))
          v (if rec (verify verify-bin (:dialect job) rec) {:outcome "no-sections" :detail ""})
          ok (= "accepted" (:outcome v))
          hint? (and (not ok) (< attempt max-repairs) (some? (repair-hint (:dialect job) v)))]
      (write-line logw {:job (:id job) :attempt attempt :usage (:usage r)
                        :latency (:latency r) :outcome (:outcome v) :detail (:detail v)
                        :actual (:actual v) :hint hint?})
      (let [first-fail (or first-fail (when (= "compile-error" (:outcome v)) {:v v :rec rec}))]
        (if (or ok (>= attempt max-repairs))
          (do (write-line raww (assoc (or rec {:id (:id job) :raw (:text r)})
                                      :verify (:outcome v) :job job))
              (when (and ok first-fail (contains? #{"write" "extend"} (:task_type job)))
                (let [broken (get-in first-fail [:v :program])
                      msg (get-in first-fail [:v :detail])]
                  (write-line raww (assoc rec :id (str (:id job) "-fix") :task_type "fix"
                                          :prompt (str "This program does not compile:\n\n```pascal\n" broken "```\n\n"
                                                       "The compiler says: " msg "\nFix it.")
                                          :broken_source broken :compiler_message msg
                                          :verify "accepted" :job job))))
              (:outcome v))
          (recur (conj messages {:role "assistant" :content [{:text (:text r)}]}
                       {:role "user" :content [{:text (repair-text (:dialect job) v prev)}]})
                 (inc attempt) first-fail v (if hint? (inc hints) hints)))))))

(defn done-jobs [out]
  (let [f (io/file (str out "/raw.jsonl"))]
    (if (.exists f)
      (set (for [l (str/split-lines (slurp f)) :when (not (str/blank? l))
                 :let [o (json/parse-string l true)]]
             (get-in o [:job :id])))
      #{})))

;; Quotas under TARGET (spec 3): the accepted non-fix records of a dialect
;; are split over the task types in proportion 50/15/10/10 (fix records
;; come on top and are cut to their share when the data set is built), and
;; no topic gets more than GEN_TOPIC_SLACK (default 1.2) times an even share,
;; so the strong topics leave room for the weak ones; the slack lets a topic
;; the generator rarely gets right fall short without blocking the dialect.
(def tt-share {"write" 50 "extend" 15 "convert" 10 "unit" 10})
(def topic-slack (parse-double (or (System/getenv "GEN_TOPIC_SLACK") "1.2")))

(defn quotas [target]
  (let [tot (reduce + (vals tt-share))
        base (into {} (for [[t s] tt-share] [t (long (Math/floor (/ (* target s) tot)))]))]
    {:tt (update base "write" + (- target (reduce + (vals base))))
     :topic (long (Math/ceil (* topic-slack (/ target (count (topics))))))}))

(defn accepted-cells [out]
  ;; accepted non-fix records counted per [:tt dialect task_type] and
  ;; [:topic dialect topic]
  (let [f (io/file out "raw.jsonl")]
    (if (.exists f)
      (frequencies (for [l (str/split-lines (slurp f)) :when (not (str/blank? l))
                         :let [o (json/parse-string l true) j (:job o)]
                         :when (and (= "accepted" (:verify o)) (not (str/ends-with? (str (:id o)) "-fix")))
                         k [[:tt (:dialect j) (:task_type j)] [:topic (:dialect j) (:topic j)]]]
                     k))
      {})))

(defn wanted? [q cells j]
  ;; nil quotas (no TARGET): every job
  (or (nil? q)
      (and (< (get cells [:tt (:dialect j) (:task_type j)] 0) (get-in q [:tt (:task_type j)] 0))
           (< (get cells [:topic (:dialect j) (:topic j)] 0) (:topic q)))))

(defn shortfall [q cells queue]
  ;; the task-type quotas still open, per dialect, with the queued jobs that
  ;; could fill them
  (for [d dialects [t n] (sort (:tt q))
        :let [have (get cells [:tt d t] 0)]
        :when (< have n)]
    {:dialect d :task_type t :have have :quota n
     :queued (count (filter #(and (= d (:dialect %)) (= t (:task_type %)) (wanted? q cells %)) queue))}))

(defn run [out verify-bin mx workers target]
  ;; WORKERS jobs at a time, each worker with its own tmp directory; at most
  ;; MAX jobs in this call; with TARGET a job is handed out only while its
  ;; dialect's task-type quota and topic cap (above) are open, counted over
  ;; all calls (jobs in progress may add up to WORKERS more); jobs passed
  ;; over stay unrun. At the end the quotas still open are printed (more
  ;; jobs: plan-more). The first failure (Converse: expired session) stops
  ;; the hand-out of new jobs, the jobs in progress finish, then exit 4.
  ;; Order of jobs in the files may differ from jobs.edn
  (let [jobs (edn/read-string (slurp (str out "/jobs.edn")))
        done (done-jobs out)
        queue (atom (vec (remove #(done (:id %)) jobs)))
        q (when target (quotas target))
        acc (atom (accepted-cells out))
        issued (atom 0)
        stop (atom nil)
        take-job #(locking queue
                    (when (< @issued (or mx Long/MAX_VALUE))
                      (let [qu @queue
                            i (first (keep-indexed (fn [i j] (when (wanted? q @acc j) i)) qu))]
                        (when i
                          (reset! queue (into (subvec qu 0 i) (subvec qu (inc i))))
                          (swap! issued inc)
                          (qu i)))))
        system (spec-text)]
    (with-open [logw (io/writer (str out "/log.jsonl") :append true)
                raww (io/writer (str out "/raw.jsonl") :append true)]
      (let [fs (doall
                (for [k (range workers)]
                  (future
                    (let [tmp (str out "/tmp/w" k)]
                      (loop []
                        (when-let [job (and (nil? @stop) (take-job))]
                          (try (let [o (run-job tmp verify-bin system job logw raww)]
                                 (when (= "accepted" o)
                                   (swap! acc #(-> % (update [:tt (:dialect job) (:task_type job)] (fnil inc 0))
                                                   (update [:topic (:dialect job) (:topic job)] (fnil inc 0)))))
                                 (locking write-lock (println (:id job) o)))
                               (catch Exception e
                                 (reset! stop (str (:id job) ": " (ex-message e)))))
                          (recur)))))))]
        (run! deref fs)))
    (when (and q (nil? @stop) (< @issued (or mx Long/MAX_VALUE)))
      (doseq [s (shortfall q @acc @queue)]
        (println (format "open: %s %s %d/%d (no job left that could fill it)" (:dialect s) (:task_type s) (:have s) (:quota s)))))
    (when @stop
      (binding [*out* *err*] (println "stopped:" @stop))
      (System/exit 4))))

(defn reverify [out verify-bin]
  ;; verifies the final records that were not accepted again (no Bedrock):
  ;; dialect, outcome, first compiler error or the actual output
  (doseq [l (str/split-lines (slurp (str out "/raw.jsonl")))
          :when (not (str/blank? l))
          :let [o (json/parse-string l true)]
          :when (and (:answer o) (not= "accepted" (:verify o)))]
    (let [v (verify verify-bin (:dialect o) o)]
      (println (json/generate-string {:id (:id o) :dialect (:dialect o) :outcome (:outcome v)
                                      :detail (:detail v) :actual (:actual v) :context (:context v)
                                      :expected (when (:actual v) (:expected_stdout o))})))))

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

(let [[cmd a b c d e] *command-line-args*]
  (case cmd
    "plan" (plan a b (parse-long (or c "1")))
    "plan-more" (plan-more a b (parse-long (or c "1")) d)
    "quota" (quota-report a (parse-long b))
    "run" (run a b (some-> c parse-long) (parse-long (or d "1")) (some-> e parse-long))
    "reverify" (reverify a b)
    "stats" (stats a)
    "fewshot" (fewshot (rest *command-line-args*))))
