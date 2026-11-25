namespace MaxDemo;

uses
  System.Globalization;

type
  Program = class
  public
    class method Max<T>(values: array of T): T; where T is IComparable<T>;
    begin
      result := values[0];
      for each v in values do
        if v.CompareTo(result) > 0 then
          result := v;
    end;

    class method Main(args: array of String): Int32;
    begin
      writeLn(Max<Integer>([7, 42, 13]));
      writeLn(Max<Double>([2.5, 0.5, 9.75]).ToString(CultureInfo.InvariantCulture));
      writeLn(Max<String>(['pear', 'apple', 'zucchini']));
    end;
  end;

end.
