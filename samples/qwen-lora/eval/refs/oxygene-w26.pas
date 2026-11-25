namespace Palin;

uses
  System.Text;

type
  Program = class
  public
    class method IsPalindrome(s: String): Boolean;
    begin
      var sb := new StringBuilder;
      for each c in s.ToLowerInvariant do
        if Char.IsLetter(c) then
          sb.Append(c);
      var t := sb.ToString;
      for i: Integer := 0 to t.Length div 2 - 1 do
        if t[i] <> t[t.Length - 1 - i] then
          exit false;
      exit true;
    end;

    class method Main(args: array of String): Int32;
    begin
      var texts: array of String := ['racecar', 'Oxygene', 'A man a plan a canal Panama', 'No lemon, no melon'];
      for each t in texts do begin
        var answer := if IsPalindrome(t) then 'yes' else 'no';
        writeLn($'{t}: {answer}');
      end;
    end;
  end;

end.
