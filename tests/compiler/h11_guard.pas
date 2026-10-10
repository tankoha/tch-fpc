{ (H11) When is an interface returned by a guard function released? }
program h11_guard;
{$mode objfpc}{$H+}
uses SysUtils;
var
  Log: String = '';
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
  WriteLn(Log);
  Log := 'discarded:';
  Discarded;
  Log := Log + ' returned';
  WriteLn(Log);
  Log := 'exception:';
  try
    WithException;
  except
    on E: Exception do Log := Log + ' caught';
  end;
  WriteLn(Log);
end.
