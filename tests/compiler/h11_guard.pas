{ (H11) When is an interface returned by a guard function released? }
program h11_guard;
{$mode objfpc}{$H+}
uses SysUtils;
var
  Log: String = '';
  Failures: Integer = 0;

{ H11 resolved: released at routine exit, at once when discarded, and
  while unwinding }
procedure Expect(const Exp: String);
begin
  WriteLn(Log);
  if Log = Exp then
    WriteLn('  ok')
  else
  begin
    WriteLn('  FAIL: expected ', Exp);
    Inc(Failures);
  end;
end;
type
  TGuard = class(TInterfacedObject)
    destructor Destroy; override;
  end;
destructor TGuard.Destroy;
begin
  Log := Log + ' [released]';
  inherited;
end;
function NoGradGuard: IInterface;
begin
  Result := TGuard.Create;
end;
procedure WithLocal;
var
  G: IInterface;
begin
  Log := Log + ' start';
  G := NoGradGuard;
  Log := Log + ' after-assign';
  begin
    Log := Log + ' inner-block';
  end;
  Log := Log + ' end-of-body';
end;
procedure Discarded;
begin
  Log := Log + ' start';
  NoGradGuard;                 // result not stored
  Log := Log + ' after-call';
  Log := Log + ' end-of-body';
end;
procedure WithException;
var
  G: IInterface;
begin
  Log := Log + ' start';
  G := NoGradGuard;
  raise Exception.Create('boom');
end;
begin
  Log := 'local:';
  WithLocal;
  Log := Log + ' returned';
  Expect('local: start after-assign inner-block end-of-body [released] returned');
  Log := 'discarded:';
  Discarded;
  Log := Log + ' returned';
  Expect('discarded: start [released] after-call end-of-body returned');
  Log := 'exception:';
  try
    WithException;
  except
    on E: Exception do Log := Log + ' caught';
  end;
  Expect('exception: start [released] caught');
  if Failures > 0 then
    Halt(1);
end.
