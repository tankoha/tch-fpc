namespace Tokens;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var s := '12 + (34 * 5) - 6';
      var i := 0;
      while i < s.Length do begin
        var c := s[i];
        if Char.IsDigit(c) then begin
          var start := i;
          while (i < s.Length) and Char.IsDigit(s[i]) do
            inc(i);
          writeLn('num ' + s.Substring(start, i - start));
          continue;
        end;
        case c of
          '+', '-', '*', '/': writeLn($'op {c}');
          '(': writeLn('lpar (');
          ')': writeLn('rpar )');
        end;
        inc(i);
      end;
    end;
  end;

end.
