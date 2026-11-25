namespace Totals;

type
  Program = class
  public
    class method Sum(values: array of Integer): Integer;
    begin
      result := 0;
      for each v in values do
        result := result + v
    end;

    class method Main(args: array of String): Int32;
    begin
      var t := Sum([1, 2, 3]);
      writeLn($'total {t}');
      exit 0;
    end;
  end;

end.
