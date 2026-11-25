namespace CharCount;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var text := 'hello world';
      var counts := new Dictionary<Char, Integer>;
      var order := new List<Char>;
      for each c in text do begin
        if c = ' ' then
          continue;
        if counts.ContainsKey(c) then
          counts[c] := counts[c] + 1
        else begin
          counts[c] := 1;
          order.Add(c);
        end;
      end;
      for each c in order do
        if counts[c] > 1 then
          writeLn($'{c}: {counts[c]}');
    end;
  end;

end.
