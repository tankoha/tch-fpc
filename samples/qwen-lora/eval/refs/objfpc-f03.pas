program Ticker;
{$mode objfpc}{$H+}
type
  TTickEvent = procedure(N: Integer) of object;

  TClock = class
    OnTick: TTickEvent;
    procedure Run;
  end;

  TPrinter = class
    procedure Print(N: Integer);
  end;

procedure TClock.Run;
var
  I: Integer;
begin
  for I := 1 to 3 do
    OnTick(I);
end;

procedure TPrinter.Print(N: Integer);
begin
  WriteLn('tick ', N);
end;

var
  C: TClock;
  P: TPrinter;
begin
  C := TClock.Create;
  P := TPrinter.Create;
  C.OnTick := @P.Print;
  C.Run;
  C.Free;
  P.Free;
end.
