namespace Numbers;

uses
  System.Globalization;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var values: array of Integer := [3, 8, 12, 5, 9];
      var sum := 0;
      var product: Int64 := 1;
      for each v in values do begin
        sum := sum + v;
        product := product * v;
      end;
      writeLn($'sum {sum}');
      writeLn($'product {product}');
      writeLn('average ' + (Double(sum) / values.Length).ToString('0.00', CultureInfo.InvariantCulture));
    end;
  end;

end.
