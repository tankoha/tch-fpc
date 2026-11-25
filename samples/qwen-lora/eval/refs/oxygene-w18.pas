namespace Parking;

uses
  System.Collections.Generic;

type
  Program = class
  public
    class method Main(args: array of String): Int32;
    begin
      var lot := new List<String>;
      var events: array of String := ['arrive A', 'arrive B', 'arrive C', 'arrive D', 'leave B', 'arrive D', 'leave A'];
      for each e in events do begin
        var parts := e.Split([' ']);
        var car := parts[1];
        if parts[0] = 'arrive' then begin
          if lot.Count = 3 then
            writeLn($'full, {car} waits away')
          else begin
            lot.Add(car);
            writeLn($'{car} parks');
          end;
        end
        else begin
          lot.Remove(car);
          writeLn($'{car} leaves');
        end;
      end;
      lot.Sort((a, b) -> String.CompareOrdinal(a, b));
      writeLn(String.Join(' ', lot));
    end;
  end;

end.
