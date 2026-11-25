namespace Evens;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var l := new List<Integer>;
      for i: Integer := 1 to 10 do
        if i mod 2 = 0 then
          l.Add(i);
      writeLn($'count {l.Count}');
      writeLn($'last {l[l.Count - 1]}');
    end;
  end;

end.
