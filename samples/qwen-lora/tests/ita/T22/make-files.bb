;; The Oxygene files of the T22 ITa (written by the harness owner; the
;; outputs are committed): four correct console programs and planted
;; faults, each <name>.pas with <name>.expected where it has one, and
;; expected.tsv (name <TAB> outcome). Usage (from the sample's root):
;; bb tests/ita/T22/make-files.bb
(require '[clojure.string :as str])
(def d "tests/ita/T22/files/")
(defn prog [ns body & [types]]
  (str "namespace " ns ";\n\ntype\n" (or types "")
       "  Program = class\n  public\n    class method Main(args: array of String): Int32;\n    begin\n"
       body "    end;\n  end;\n\nend.\n"))
(def files
  [["counter" (prog "Counter"
                    "      var c := new Counter;\n      for i: Integer := 1 to 5 do\n        c.Add(i);\n      writeLn(c.Total);\n      writeLn(c.Count);\n"
                    "  Counter = class\n  public\n    property Total: Integer read private write;\n    property Count: Integer read private write;\n    method Add(aValue: Integer);\n    begin\n      Total := Total + aValue;\n      Count := Count + 1;\n    end;\n  end;\n\n")
    "15\n5\n" "accepted"]
   ["words" (prog "Words"
                  "      var list := new List<String>;\n      list.Add('pear');\n      list.Add('apple');\n      list.Add('fig');\n      list.Sort;\n      for each w in list do\n        writeLn(w);\n      writeLn(list.Count);\n")
    "apple\nfig\npear\n3\n" "accepted"]
   ["guard" (prog "Guard"
                  "      try\n        raise new ArgumentException('bad value');\n      except\n        on E: ArgumentException do\n          writeLn('caught: ' + E.Message);\n      end;\n      writeLn('done');\n")
    "caught: bad value\ndone\n" "accepted"]
   ["fib" (prog "Fib"
                "      var a := 0;\n      var b := 1;\n      for i: Integer := 1 to 10 do begin\n        var t := a + b;\n        a := b;\n        b := t;\n      end;\n      writeLn(a);\n")
    "55\n" "accepted"]
   ["x-compile" (prog "Broken" "      writeLn(Missing);\n") "x\n" "compile-error"]
   ["x-output" (prog "Wrong" "      writeLn(6 * 7);\n") "41\n" "wrong-output"]
   ["x-run" (prog "Code" "      writeLn('before');\n      exit 4;\n") "before\n" "run-error"]
   ["x-pitfall" (prog "Pit" "      writeLn(Missing); // expect-error: Unknown identifier \"Missing\"\n") nil "accepted"]
   ["x-pitfall-wrong" (prog "Pit2" "      writeLn(Missing); // expect-error: Division by zero\n") nil "wrong-error"]
   ["x-no-expected" (prog "Free" "      writeLn('free');\n") nil "no-expected"]
   ["x-fpc" "program NotOxygene;\nbegin\n  WriteLn('fpc');\nend.\n" "fpc\n" "compile-error"]])
(.mkdirs (java.io.File. d))
(doseq [[n src exp _] files]
  (spit (str d n ".pas") src)
  (when exp (spit (str d n ".expected") exp)))
(spit "tests/ita/T22/expected.tsv" (str/join (for [[n _ _ o] files] (str n "\t" o "\n"))))
(println (count files) "files")
