namespace Counter;

type
  Counter = class
  public
    property Total: Integer read private write;
    property Count: Integer read private write;
    method Add(aValue: Integer);
    begin
      Total := Total + aValue;
      Count := Count + 1;
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var c := new Counter;
      for i: Integer := 1 to 5 do
        c.Add(i);
      writeLn(c.Total);
      writeLn(c.Count);
    end;
  end;

end.
