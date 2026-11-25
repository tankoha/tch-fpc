namespace Pairs;

type
  Pair<T> = public class
  public
    property First: T;
    property Second: T;
    constructor(aFirst, aSecond: T);
    begin
      First := aFirst;
      Second := aSecond;
    end;
    method Swap: Pair<T>;
    begin
      result := new Pair<T>(Second, First);
    end;
    method ToString: String; override;
    begin
      result := $'({First}, {Second})';
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var p := new Pair<Integer>(1, 2);
      var q := new Pair<String>('left', 'right');
      writeLn(p.Swap.ToString);
      writeLn(q.Swap.ToString);
    end;
  end;

end.
