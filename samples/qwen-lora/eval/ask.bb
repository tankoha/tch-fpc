;; Records the replies of an Ollama model to the evaluation tasks (eval/
;; tasks.jsonl, or another tasks file of the same form), as DESIGN
;; 8.17 "Runs" says: Ollama's chat endpoint (the model's own renderer and
;; parser), the deployed sampling parameters, two fixed seeds per task,
;; thinking on and off; a regression tool-call task is given the four tools
;; below. One line per reply: {"task","seed","think","content",
;; "tool_calls"}. Written by the harness owner; the user runs the same
;; script for the "before" and "after" evaluations (M3, M5).
;; Usage: bb ask.bb MODEL TASKS OUT [NUM_PREDICT]
(require '[babashka.http-client :as http] '[cheshire.core :as json]
         '[clojure.java.io :as io] '[clojure.string :as str])
(defn tool [name desc props]
  {:type "function"
   :function {:name name :description desc
              :parameters {:type "object" :properties (into {} (for [p props] [(keyword p) {:type "string"}]))
                           :required (vec props)}}})
(def tools
  [(tool "get_weather" "The current weather of a city." ["city"])
   (tool "convert_currency" "Converts an amount between two currencies at the current rate." ["amount" "from" "to"])
   (tool "get_time" "The current local time in a city or time zone." ["place"])
   (tool "search_web" "Searches the web and returns the top results." ["query"])])
(let [[model tasks out np] *command-line-args*
      np (parse-long (or np "1024"))]
  (with-open [w (io/writer out)]
    (doseq [l (line-seq (io/reader tasks)) :when (not (str/blank? l))
            :let [t (json/parse-string l true)]
            think [true false] seed [1 2]]
      (let [body (cond-> {:model model :stream false :think think
                          :messages [{:role "user" :content (:prompt t)}]
                          :options {:seed seed :num_predict np}}
                   (= (:check t) "tool-call") (assoc :tools tools))
            r (-> (http/post "http://127.0.0.1:11434/api/chat"
                             {:headers {"Content-Type" "application/json"}
                              :body (json/generate-string body) :timeout 600000})
                  :body (json/parse-string true))
            m (:message r)]
        (.write w (str (json/generate-string
                        {:task (:id t) :seed seed :think think :content (or (:content m) "")
                         :tool_calls (mapv (fn [c] {:name (get-in c [:function :name])
                                                    :arguments (get-in c [:function :arguments])})
                                           (:tool_calls m))})
                       "\n"))
        (.flush w)))))
