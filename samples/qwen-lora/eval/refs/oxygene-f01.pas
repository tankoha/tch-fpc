namespace Counting;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var list := new List<String>;
      list.Add('a');
      list.Add('b');
      list.Add('c');
      writeLn($'count {list.Count}');
    end;
  end;

end.
