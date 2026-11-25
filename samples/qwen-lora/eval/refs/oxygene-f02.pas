namespace Finder;

type
  Finder = public class
  public
    class method IndexOf<T>(items: array of T; item: T): Integer;
    begin
      for i: Integer := 0 to items.Length - 1 do
        if items[i].Equals(item) then
          exit i;
      exit -1;
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var idx := Finder.IndexOf<String>(['x', 'y', 'z'], 'z');
      writeLn($'found at {idx}');
    end;
  end;

end.
