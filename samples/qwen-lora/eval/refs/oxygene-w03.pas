namespace Words;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var words := new List<String>(['kiwi', 'banana', 'fig', 'cherry', 'apple']);
      var sorted := new List<String>(words);
      sorted.Sort((a, b) -> String.CompareOrdinal(a, b));
      writeLn(String.Join(',', sorted));
      var long := new List<String>;
      var longest := 0;
      for each w in words do begin
        if w.Length > 4 then
          long.Add(w);
        if w.Length > longest then
          longest := w.Length;
      end;
      writeLn(String.Join(' ', long));
      writeLn($'longest {longest}');
    end;
  end;

end.
