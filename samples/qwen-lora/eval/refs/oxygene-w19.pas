namespace Triangles;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var tri: block(n: Integer): Integer := n -> n * (n + 1) div 2;
      var list := new List<Integer>;
      for i: Integer := 1 to 8 do
        list.Add(tri(i));
      writeLn(String.Join(' ', list));
    end;
  end;

end.
