namespace Thermo;

type
  Thermometer = public class
  private
    fValue: Integer;
  public
    event Changed: block(aOld, aNew: Integer);
    method SetValue(aValue: Integer);
    begin
      if aValue = fValue then
        exit;
      var prev := fValue;
      fValue := aValue;
      if assigned(Changed) then
        Changed(prev, aValue);
    end;
  end;

  Logger = public class
  public
    method Log(aOld, aNew: Integer);
    begin
      writeLn($'logged {aNew}');
    end;
  end;

  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var t := new Thermometer;
      var log := new Logger;
      t.Changed += (o, n) -> writeLn($'{o} -> {n}');
      t.Changed += @log.Log;
      t.SetValue(20);
      t.SetValue(25);
      t.SetValue(25);
      t.Changed -= @log.Log;
      t.SetValue(18);
    end;
  end;

end.
