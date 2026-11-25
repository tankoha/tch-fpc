namespace HexStack;

uses
  System.Collections.Generic;

type
  SimpleStack<T> = public class
  private
    fItems: List<T> := new List<T>;
  public
    method Push(aValue: T);
    begin
      fItems.Add(aValue);
    end;
    method Pop: T;
    begin
      result := fItems[fItems.Count - 1];
      fItems.RemoveAt(fItems.Count - 1);
    end;
    method Peek: T;
    begin
      result := fItems[fItems.Count - 1];
    end;
    property Count: Integer read fItems.Count;
  end;

  Program = class
  public
    class method ToHex(n: Integer): String;
    begin
      var digits := '0123456789ABCDEF';
      var s := new SimpleStack<Char>;
      repeat
        s.Push(digits[n mod 16]);
        n := n div 16;
      until n = 0;
      result := '';
      while s.Count > 0 do
        result := result + s.Pop;
    end;

    class method Main(args: array of String): Int32;
    begin
      for each n in [10, 255, 4096] do
        writeLn($'{n} = {ToHex(n)}');
    end;
  end;

end.
