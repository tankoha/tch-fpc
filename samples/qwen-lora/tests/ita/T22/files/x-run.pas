namespace Code;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      writeLn('before');
      exit 4;
    end;
  end;

end.
