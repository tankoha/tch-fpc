namespace Pairs;

type
  Program = class
  public
    class method SumDiff(a, b: Integer): tuple of (Integer, Integer);
    begin
      result := (a + b, a - b);
    end;

    class method Main(args: array of String): Int32;
    begin
      var (s, d) := SumDiff(10, 4);
      writeLn($'sum={s} diff={d}');
    end;
  end;

end.
