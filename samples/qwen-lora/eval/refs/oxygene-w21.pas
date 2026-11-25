namespace Square;

uses
  System.Text;

type
  Program = class
  public
    class method Draw(n: Integer): String;
    begin
      var sb := new StringBuilder;
      for r: Integer := 0 to n - 1 do begin
        for c: Integer := 0 to n - 1 do
          if (r = 0) or (r = n - 1) or (c = 0) or (c = n - 1) then
            sb.Append('*')
          else
            sb.Append(' ');
        sb.Append(#10);
      end;
      result := sb.ToString;
    end;

    class method Main(args: array of String): Int32;
    begin
      write(Draw(5));
    end;
  end;

end.
