;; Builds the synthetic raw set of the T19 ITa (written by the harness
;; owner; the outputs are committed, so the ITa does not change when this
;; file is rerun): 16 small correct programs in four dialects (objfpc,
;; FPC delphi, Unleashed, Delphi) plus records with planted faults, the
;; evaluation tasks the leak check compares with, two example files, and
;; the expected outcome of every record (id <TAB> accepted | reject reason
;; up to its colon). Usage (from the sample's root): bb tests/ita/T19/make-raw.bb
(require '[clojure.string :as str] '[cheshire.core :as json])

(def d "tests/ita/T19/")
(defn fence [& srcs] (str/join "\n" (map #(str "```pascal\n" % "```") srcs)))
(def gen {:model "synthetic" :version "0" :date "2026-11-25" :prompt_id "t19" :temperature 0 :repair_attempts 0})
(defn rec [id dialect tt prompt srcs expected & {:as extra}]
  (merge {:id id :dialect dialect :task_type tt :topic "test" :features [] :language "en"
          :prompt prompt :answer (str "Here is the program.\n\n" (apply fence srcs) "\n")
          :expected_stdout expected :generator gen}
         extra))

;; ---- the good programs ----
(def o1 "program SumSq;\n{$mode objfpc}{$H+}\nvar\n  I, S: Integer;\nbegin\n  S := 0;\n  for I := 1 to 10 do\n    S := S + I * I;\n  WriteLn(S);\nend.\n")
(def o2 "program FizzBuzz;\n{$mode objfpc}{$H+}\nvar\n  N: Integer;\nbegin\n  for N := 1 to 15 do\n    if N mod 15 = 0 then WriteLn('FizzBuzz')\n    else if N mod 3 = 0 then WriteLn('Fizz')\n    else if N mod 5 = 0 then WriteLn('Buzz')\n    else WriteLn(N);\nend.\n")
(def o2-out "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n")
(def o3 "program Rev;\n{$mode objfpc}{$H+}\nfunction Reverse(const S: string): string;\nvar\n  K: Integer;\nbegin\n  Result := '';\n  for K := Length(S) downto 1 do\n    Result := Result + S[K];\nend;\n\nbegin\n  WriteLn(Reverse('pascal'));\nend.\n")
(def o4 "program Sieve;\n{$mode objfpc}{$H+}\nconst\n  Max = 30;\nvar\n  Composite: array[2..Max] of Boolean;\n  P, Q: Integer;\nbegin\n  for P := 2 to Max do Composite[P] := False;\n  for P := 2 to Max do\n    if not Composite[P] then\n    begin\n      Write(P, ' ');\n      Q := P * P;\n      while Q <= Max do\n      begin\n        Composite[Q] := True;\n        Q := Q + P;\n      end;\n    end;\n  WriteLn;\nend.\n")
(def o5 "program Counter;\n{$mode objfpc}{$H+}\ntype\n  TCounter = class\n  private\n    FValue: Integer;\n  public\n    procedure Bump;\n    property Value: Integer read FValue;\n  end;\n\nprocedure TCounter.Bump;\nbegin\n  Inc(FValue, 2);\nend;\n\nvar\n  C: TCounter;\nbegin\n  C := TCounter.Create;\n  try\n    C.Bump; C.Bump; C.Bump;\n    WriteLn('value=', C.Value);\n  finally\n    C.Free;\n  end;\nend.\n")
(def o6 "program Dist;\n{$mode objfpc}{$H+}\nuses\n  SysUtils;\ntype\n  TPoint2 = record\n    X, Y: Double;\n  end;\n\nfunction Distance(const A, B: TPoint2): Double;\nbegin\n  Result := Sqrt(Sqr(A.X - B.X) + Sqr(A.Y - B.Y));\nend;\n\nvar\n  A, B: TPoint2;\nbegin\n  A.X := 0; A.Y := 0;\n  B.X := 3; B.Y := 4;\n  WriteLn(FormatFloat('0.00', Distance(A, B)));\nend.\n")
(def o7 "program Gcd;\n{$mode objfpc}{$H+}\nfunction GreatestDivisor(A, B: LongInt): LongInt;\nbegin\n  while B <> 0 do\n  begin\n    Result := B;\n    B := A mod B;\n    A := Result;\n  end;\n  Result := A;\nend;\n\nbegin\n  WriteLn(GreatestDivisor(84, 36));\nend.\n")
(def o8u "unit CubeMath;\n{$mode objfpc}{$H+}\ninterface\nfunction Cube(X: Integer): Integer;\nimplementation\nfunction Cube(X: Integer): Integer;\nbegin\n  Result := X * X * X;\nend;\nend.\n")
(def o8 "program UseCube;\n{$mode objfpc}{$H+}\nuses\n  CubeMath;\nbegin\n  WriteLn('cube=', Cube(3));\nend.\n")
(def d1 "program Fact;\n{$mode delphi}\nfunction Factorial(N: Integer): Int64;\nbegin\n  if N <= 1 then\n    Result := 1\n  else\n    Result := N * Factorial(N - 1);\nend;\n\nbegin\n  WriteLn(Factorial(5));\nend.\n")
(def d2 "program Sorted;\n{$mode delphi}\nuses\n  Classes;\nvar\n  L: TStringList;\n  I: Integer;\nbegin\n  L := TStringList.Create;\n  try\n    L.Add('pear'); L.Add('apple'); L.Add('fig');\n    L.Sort;\n    for I := 0 to L.Count - 1 do\n      WriteLn(L[I]);\n  finally\n    L.Free;\n  end;\nend.\n")
(def d3 "program Fib;\n{$mode delphi}\nvar\n  A, B, T, I: Integer;\nbegin\n  A := 0; B := 1;\n  for I := 1 to 10 do\n  begin\n    Write(A, ',');\n    T := A + B; A := B; B := T;\n  end;\n  WriteLn;\nend.\n")
(def d4 "program Vowels;\n{$mode delphi}\nfunction CountVowels(const S: string): Integer;\nvar\n  Ch: Char;\nbegin\n  Result := 0;\n  for Ch in S do\n    if Ch in ['a', 'e', 'i', 'o', 'u'] then\n      Inc(Result);\nend;\n\nbegin\n  WriteLn(CountVowels('programming in pascal'));\nend.\n")
(def u1 "program Squares;\n{$mode unleashed}\nbegin\n  for var I := 1 to 4 do\n    WriteLn(I * I);\nend.\n")
(def u2 "program Upper;\n{$mode unleashed}\nuses\n  SysUtils;\nbegin\n  var S := 'hello world';\n  WriteLn(UpperCase(S));\nend.\n")
(def p1 "program Greeting;\n{$APPTYPE CONSOLE}\nuses\n  System.SysUtils;\nbegin\n  WriteLn(Format('%s, %d!', ['Hello', 42]));\nend.\n")
(def p2 "program Triangle;\n{$APPTYPE CONSOLE}\nvar\n  Row, Col: Integer;\nbegin\n  for Row := 1 to 3 do\n  begin\n    for Col := 1 to Row do\n      Write('*');\n    WriteLn;\n  end;\nend.\n")

(def good
  [(rec "o1" "objfpc" "write" "Print the sum of the squares of 1 to 10." [o1] "385\n")
   (rec "o2" "objfpc" "write" "Print FizzBuzz for 1 to 15." [o2] o2-out)
   (rec "o3" "objfpc" "write" "Reverse the word pascal." [o3] "lacsap\n")
   (rec "o4" "objfpc" "write" "Print the primes up to 30 with a sieve." [o4] "2 3 5 7 11 13 17 19 23 29 \n")
   (rec "o5" "objfpc" "write" "A counter class bumped three times by two." [o5] "value=6\n")
   (rec "o6" "objfpc" "write" "The distance between two points." [o6] "5.00\n")
   (rec "o7" "objfpc" "fix" "Fix this greatest common divisor program." [o7] "12\n")
   (rec "o8" "objfpc" "unit" "A unit with a cube function and a program using it." [o8u o8] "cube=27\n")
   (rec "d1" "fpc-delphi" "write" "Print 5 factorial." [d1] "120\n")
   (rec "d2" "fpc-delphi" "write" "Sort three fruit names." [d2] "apple\nfig\npear\n")
   (rec "d3" "fpc-delphi" "write" "Print the first ten Fibonacci numbers." [d3] "0,1,1,2,3,5,8,13,21,34,\n")
   (rec "d4" "fpc-delphi" "write" "Count the vowels of a phrase." [d4] "6\n")
   (rec "u1" "unleashed" "write" "Print the squares of 1 to 4." [u1] "1\n4\n9\n16\n")
   (rec "u2" "unleashed" "write" "Upper-case hello world." [u2] "HELLO WORLD\n")
   (rec "p1" "delphi" "write" "Greet with a formatted string." [p1] "Hello, 42!\n")
   (rec "p2" "delphi" "write" "Print a triangle of stars." [p2] "*\n**\n***\n")])

;; ---- the evaluation tasks and the example files ----
(def e1-ref "program Collatz;\n{$mode objfpc}{$H+}\nvar\n  N, Steps: Integer;\nbegin\n  N := 27;\n  Steps := 0;\n  while N <> 1 do\n  begin\n    if Odd(N) then N := 3 * N + 1 else N := N div 2;\n    Inc(Steps);\n  end;\n  WriteLn(Steps);\nend.\n")
(def e2-prompt "Write a program that prints the multiplication table of seven from one to ten, one product per line.")
(def e2-ref "program Seven;\n{$mode objfpc}{$H+}\nvar\n  K: Integer;\nbegin\n  for K := 1 to 10 do\n    WriteLn(7 * K);\nend.\n")
(def evals [{:id "eval-collatz" :prompt "Print the number of Collatz steps of 27." :reference e1-ref}
            {:id "eval-seven" :prompt e2-prompt :reference e2-ref}])
(def ex1 "program Stack;\n{$mode objfpc}{$H+}\nvar\n  Items: array[0..9] of Integer;\n  Top, V: Integer;\nbegin\n  Top := 0;\n  for V := 1 to 5 do\n  begin\n    Items[Top] := V * 10;\n    Inc(Top);\n  end;\n  while Top > 0 do\n  begin\n    Dec(Top);\n    Write(Items[Top], ' ');\n  end;\n  WriteLn;\nend.\n")
(def ex2 "program Temps;\n{$mode objfpc}{$H+}\nvar\n  C: Integer;\nbegin\n  C := -10;\n  while C <= 40 do\n  begin\n    WriteLn(C, ' C = ', C * 9 div 5 + 32, ' F');\n    C := C + 25;\n  end;\nend.\n")

;; ---- the planted faults (id, record or raw line, expected reason) ----
(def o1-spaced (str/replace o1 "  S := 0;\n" "  S := 0;   { start }\n\n"))
(def o2-near (str/replace o2 "for N := 1 to 15 do" "for N := 1 to 16 do"))
(def bad-compile "program Broken;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn(Undeclared);\nend.\n")
(def bad-out "program Off;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn(6 * 7);\nend.\n")
(def bad-run "program Halts;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn('before');\n  Halt(3);\nend.\n")
(def tok-src "program TokenCase;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn('tokens');\nend.\n")
(def oxy "namespace Hello;\n\ntype\n  Program = class\n  public\n    class method Main(args: array of String): Int32;\n    begin\n      writeLn('Hello');\n    end;\n  end;\n\nend.\n")

(def planted
  [[(rec "x-dup" "objfpc" "write" "Sum the squares from one to ten." [o1] "385\n") "duplicate"]
   [(rec "x-spaced" "objfpc" "write" "Sum of squares, again." [o1-spaced] "385\n") "duplicate"]
   [(rec "x-near" "objfpc" "write" "FizzBuzz once more." [o2-near] (str o2-out "16\n")) "near-duplicate"]
   [(rec "x-leak-ref" "objfpc" "write" "Count the steps of a famous sequence." [e1-ref] "111\n") "leak-eval"]
   [(rec "x-leak-prompt" "objfpc" "write" e2-prompt
         ["program Table7;\n{$mode objfpc}{$H+}\nvar\n  Factor: Integer;\nbegin\n  Factor := 1;\n  repeat\n    WriteLn(Factor * 7);\n    Inc(Factor);\n  until Factor > 10;\nend.\n"]
         "7\n14\n21\n28\n35\n42\n49\n56\n63\n70\n") "leak-eval"]
   [(rec "x-leak-example" "objfpc" "write" "A small stack." [ex1] "50 40 30 20 10 \n") "leak-example"]
   [(rec "x-compile" "objfpc" "fix" "A broken program." [bad-compile] "") "verify"]
   [(rec "x-output" "objfpc" "write" "Print the answer." [bad-out] "41\n") "verify"]
   [(rec "x-run" "objfpc" "write" "Stop with an exit code." [bad-run] "before\n") "verify"]
   [(rec "x-dialect" "objfpc" "write" "Factorial in the wrong mode." [(str/replace d1 "Factorial(5)" "Factorial(6)")] "720\n") "dialect"]
   [(rec "x-tokenizer" "objfpc" "write" "Print the word tokens." [tok-src] "tokens\n") "tokenizer"]
   [(rec "o1" "objfpc" "write" "The same id again." ["program Other;\n{$mode objfpc}{$H+}\nbegin\n  WriteLn('other id');\nend.\n"] "other id\n") "duplicate-id"]
   [(dissoc (rec "x-no-generator" "objfpc" "write" "No generator." [o3] "lacsap\n") :generator) "schema"]
   [(assoc (rec "x-bad-dialect" "objfpc" "write" "Unknown dialect." [o3] "lacsap\n") :dialect "turbo") "schema"]
   [(assoc (rec "x-no-fence" "objfpc" "write" "No fence." [o3] "lacsap\n") :answer "program X; begin end.") "schema"]
   [(rec "x-oxygene" "oxygene" "write" "Hello in Oxygene." [oxy] "Hello\n") "verify"]
   [:bad-json "schema"]])

(defn line [r] (if (= r :bad-json) "{\"id\": \"x-bad-json\", " (json/generate-string r)))

;; the good records first, the faults after (a duplicate keeps the earlier)
(spit (str d "raw.jsonl")
      (str (str/join "\n" (concat (map line good) (map (comp line first) planted))) "\n"))
(spit (str d "eval.jsonl") (str (str/join "\n" (map json/generate-string evals)) "\n"))
(spit (str d "examples/stack.pas") ex1)
(spit (str d "examples/temps.pas") ex2)
(spit (str d "expected.tsv")
      (str (str/join "\n" (concat (map #(str (:id %) "\taccepted") good)
                                  (map (fn [[r why]] (str (if (= r :bad-json) "x-bad-json" (:id r)) "\t" why)) planted)))
           "\n"))
(println (count good) "good," (count planted) "planted")
