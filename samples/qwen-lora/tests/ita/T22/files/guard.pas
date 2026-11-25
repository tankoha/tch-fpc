namespace Guard;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      try
        raise new ArgumentException('bad value');
      except
        on E: ArgumentException do
          writeLn('caught: ' + E.Message);
      end;
      writeLn('done');
    end;
  end;

end.
