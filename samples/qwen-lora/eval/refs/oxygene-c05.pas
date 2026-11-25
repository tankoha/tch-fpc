namespace Apply;

type
  Program = class
  public
    class method ApplyAll(f: block(x: Integer): Integer);
    begin
      for i: Integer := 1 to 3 do
        writeLn(f(i));
    end;

    class method Main(args: array of String): Int32;
    begin
      ApplyAll(x -> x * 10);
    end;
  end;

end.
