namespace Actions;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method RepeatN(n: Integer; aAction: block(i: Integer));
    begin
      for i: Integer := 1 to n do
        aAction(i);
    end;

    class method Main(args: array of String): Int32;
    begin
      var list := new List<Integer>;
      RepeatN(4, method(i: Integer); begin
        list.Add(i * i);
      end);
      writeLn(String.Join(' ', list));
      RepeatN(3, i -> writeLn($'tick {i}'));
    end;
  end;

end.
