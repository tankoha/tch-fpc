namespace Parse;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      for each s in ['42', 'x'] do begin
        var value: Integer;
        if Int32.TryParse(s, out value) then
          writeLn($'parsed {value}')
        else
          writeLn('bad input');
      end;
    end;
  end;

end.
