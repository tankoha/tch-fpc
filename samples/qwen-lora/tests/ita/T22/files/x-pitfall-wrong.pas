namespace Pit2;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      writeLn(Missing); // expect-error: Division by zero
    end;
  end;

end.
