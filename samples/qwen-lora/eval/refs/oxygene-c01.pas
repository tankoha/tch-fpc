namespace Squares;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      for i: Integer := 1 to 5 do
        writeLn($'{i} squared is {i * i}');
    end;
  end;

end.
