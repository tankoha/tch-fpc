namespace Dice;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var doubles := 0;
      var total := 0;
      for each (a, b) in [(3, 4), (6, 6), (1, 2), (5, 5), (2, 6)] do begin
        writeLn($'{a}+{b}={a + b}');
        if a = b then
          inc(doubles);
        total := total + a + b;
      end;
      writeLn($'doubles {doubles}');
      writeLn($'total {total}');
    end;
  end;

end.
