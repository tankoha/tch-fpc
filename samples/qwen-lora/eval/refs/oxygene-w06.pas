namespace LinqDemo;

uses
  System.Linq;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var nums := Enumerable.Range(1, 20);
      var squares := nums.Where(n -> (n mod 2 = 1) and (n > 10)).Select(n -> n * n);
      writeLn(String.Join(' ', squares));
      writeLn($'even sum {nums.Where(n -> n mod 2 = 0).Sum}');
      writeLn($'div3 {nums.Count(n -> n mod 3 = 0)}');
    end;
  end;

end.
