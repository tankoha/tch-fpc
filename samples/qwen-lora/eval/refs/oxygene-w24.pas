namespace Reduce;

type
  Program = class
  public
    class method Gcd(a, b: Integer): Integer;
    begin
      while b <> 0 do begin
        var t := a mod b;
        a := b;
        b := t;
      end;
      result := a;
    end;

    class method Main(args: array of String): Int32;
    begin
      for each (n, d) in [(18, 24), (35, 49), (100, 75)] do begin
        var g := Gcd(n, d);
        writeLn($'{n}/{d} = {n div g}/{d div g}');
      end;
    end;
  end;

end.
