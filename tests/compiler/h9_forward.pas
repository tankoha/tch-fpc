{ (H9) Virtual abstract Forward with a default parameter repeated in the
  override; called through base and derived references. }
program h9_forward;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    H: IInterface;
    V: Double;
  end;
  TModule = class
    function Forward(const X: TT; Train: Boolean = False): TT; virtual; abstract;
  end;
  TLinear = class(TModule)
    function Forward(const X: TT; Train: Boolean = False): TT; override;
  end;
  TDropout = class(TModule)
    function Forward(const X: TT; Train: Boolean): TT; override;  // no default
  end;
function TLinear.Forward(const X: TT; Train: Boolean): TT;
begin
  Result.V := X.V * 2;
  if Train then Result.V := Result.V + 1000;
end;
function TDropout.Forward(const X: TT; Train: Boolean): TT;
begin
  Result.V := X.V;
  if Train then Result.V := 0;
end;
var
  M: TModule;
  L: TLinear;
  D: TDropout;
  X: TT;
begin
  X.V := 3;
  L := TLinear.Create;
  M := L;
  WriteLn('base M.Forward(X)          = ', M.Forward(X).V:0:1);
  WriteLn('base M.Forward(X, True)    = ', M.Forward(X, True).V:0:1);
  WriteLn('derived L.Forward(X)       = ', L.Forward(X).V:0:1);
  D := TDropout.Create;
  M := D;
  WriteLn('base M.Forward(X) on TDropout (override without default) = ',
    M.Forward(X).V:0:1);
  L.Free; D.Free;
end.
