namespace Dates;

uses
  System.Globalization;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var d := new DateTime(2024, 2, 10).AddDays(100);
      writeLn(d.ToString('yyyy-MM-dd', CultureInfo.InvariantCulture));
      writeLn(new DateTime(2024, 12, 31).DayOfWeek.ToString);
    end;
  end;

end.
