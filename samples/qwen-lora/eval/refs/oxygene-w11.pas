namespace Primes;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method IsPrime(n: Integer): Boolean;
    begin
      if n < 2 then
        exit false;
      var d := 2;
      while d * d <= n do begin
        if n mod d = 0 then
          exit false;
        inc(d);
      end;
      exit true;
    end;

    class method Main(args: array of String): Int32;
    begin
      var found := new List<Integer>;
      for i: Integer := 50 to 100 do
        if IsPrime(i) then
          found.Add(i);
      writeLn(String.Join(' ', found));
      writeLn($'count {found.Count}');
    end;
  end;

end.
