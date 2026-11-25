namespace Pit;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      writeLn(Missing); // expect-error: Unknown identifier "Missing"
    end;
  end;

end.
