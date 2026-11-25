namespace Grouping;

uses
  System.Linq;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var words: array of String := ['ant', 'bee', 'cat', 'eagle', 'fox', 'horse', 'owl', 'tiger'];
      for each g in words.GroupBy(w -> w.Length).OrderBy(g -> g.Key) do begin
        var names := String.Join(', ', g);
        writeLn($'{g.Key}: {names}');
      end;
    end;
  end;

end.
