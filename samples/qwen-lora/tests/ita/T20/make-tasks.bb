;; The evaluation tasks of the T20 ITa (written by the harness owner; the
;; output is committed): eleven small Pascal tasks in four dialects (their
;; expected output checked by compiling the reference solutions with
;; qlora_verify) and three regression tasks. They are a fixture for the
;; report, not the 200 tasks of DESIGN 8.17 (M3).
;; Usage (from the sample's root): bb tests/ita/T20/make-tasks.bb
(require '[clojure.string :as str] '[cheshire.core :as json])
(def d "tests/ita/T20/")
(defn mode [dialect]
  (case dialect
    "objfpc" "Free Pascal with {$mode objfpc}{$H+} as its second line"
    "fpc-delphi" "Free Pascal with {$mode delphi} as its second line"
    "unleashed" "Unleashed Pascal with {$mode unleashed} as its second line"
    "delphi" "Delphi with {$APPTYPE CONSOLE} as its second line"))
(defn task [id dialect what out ref]
  {:id id :dialect dialect
   :prompt (str "Write a complete console program in " (mode dialect) " that " what
                " Reply with the whole program in one ```pascal block.")
   :expected_stdout out :reference ref})
(def tasks
  [(task "o-sum" "objfpc" "prints the sum of the integers from 1 to 100 on one line." "5050\n"
         "program Sum;\n{$mode objfpc}{$H+}\nvar\n  I, S: Integer;\nbegin\n  S := 0;\n  for I := 1 to 100 do\n    S := S + I;\n  WriteLn(S);\nend.\n")
   (task "o-hello" "objfpc" "prints exactly the line Hello, World!" "Hello, World!\n"
         "program Hello;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn('Hello, World!');\nend.\n")
   (task "o-count" "objfpc" "prints the numbers 1 to 5, one per line." "1\n2\n3\n4\n5\n"
         "program Count;\n{$mode objfpc}{$H+}\nvar\n  I: Integer;\nbegin\n  for I := 1 to 5 do\n    WriteLn(I);\nend.\n")
   (task "o-max" "objfpc" "prints the largest of the numbers 3, 17 and 8." "17\n"
         "program Largest;\n{$mode objfpc}{$H+}\nuses\n  Math;\nbegin\n  WriteLn(Max(Max(3, 17), 8));\nend.\n")
   (task "d-square" "fpc-delphi" "prints the square of 12." "144\n"
         "program Square;\n{$mode delphi}\nbegin\n  WriteLn(12 * 12);\nend.\n")
   (task "d-upper" "fpc-delphi" "prints the word pascal in upper case." "PASCAL\n"
         "program Upper;\n{$mode delphi}\nuses\n  SysUtils;\nbegin\n  WriteLn(UpperCase('pascal'));\nend.\n")
   (task "d-even" "fpc-delphi" "prints the even numbers from 2 to 10 separated by single spaces on one line." "2 4 6 8 10\n"
         "program Even;\n{$mode delphi}\nvar\n  I: Integer;\nbegin\n  I := 2;\n  Write(I);\n  while I < 10 do\n  begin\n    I := I + 2;\n    Write(' ', I);\n  end;\n  WriteLn;\nend.\n")
   (task "u-hello" "unleashed" "prints exactly the line Hello from Unleashed" "Hello from Unleashed\n"
         "program Hello;\n{$mode unleashed}\nbegin\n  WriteLn('Hello from Unleashed');\nend.\n")
   (task "u-cube" "unleashed" "prints the cube of 4." "64\n"
         "program Cube;\n{$mode unleashed}\nbegin\n  var N := 4;\n  WriteLn(N * N * N);\nend.\n")
   (task "p-hello" "delphi" "prints exactly the line Hello, Delphi" "Hello, Delphi\n"
         "program Hello;\n{$APPTYPE CONSOLE}\nbegin\n  WriteLn('Hello, Delphi');\nend.\n")
   (task "p-len" "delphi" "prints the length of the string 'abcdef'." "6\n"
         "program Len;\n{$APPTYPE CONSOLE}\nbegin\n  WriteLn(Length('abcdef'));\nend.\n")
   {:id "r-tool" :dialect "regression" :prompt "What is the weather in Tokyo right now? Use the tool."
    :check "tool-call" :tool "get_weather"}
   {:id "r-capital" :dialect "regression" :prompt "What is the capital of France? Answer in one word."
    :check "contains" :text "Paris"}
   {:id "r-shell" :dialect "regression" :prompt "Give the shell command that lists the files of the current directory, in one code block."
    :check "contains" :text "ls"}])
(spit (str d "tasks.jsonl") (str (str/join "\n" (map json/generate-string tasks)) "\n"))
(doseq [t tasks :when (:reference t)]
  (spit (str d "refs/" (:id t) (if (= (:dialect t) "delphi") ".dpr" ".pas")) (:reference t))
  (spit (str d "refs/" (:id t) ".expected") (:expected_stdout t)))
(println (count tasks) "tasks")
