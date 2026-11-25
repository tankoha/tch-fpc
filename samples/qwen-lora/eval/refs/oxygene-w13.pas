namespace Loops;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var down := new List<Integer>;
      for i: Integer := 100 downto 0 step 25 do
        down.Add(i);
      writeLn(String.Join(' ', down));
      var evens := new List<Integer>;
      for i: Integer := 2 to 10 step 2 do
        evens.Add(i);
      writeLn(String.Join(' ', evens));
      var n := 0;
      var sum := 0;
      loop begin
        inc(n);
        sum := sum + n;
        if sum > 50 then
          break;
      end;
      writeLn($'stopped at {n} sum {sum}');
    end;
  end;

end.
