namespace Fib;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var a := 0;
      var b := 1;
      for i: Integer := 1 to 10 do begin
        var t := a + b;
        a := b;
        b := t;
      end;
      writeLn(a);
    end;
  end;

end.
