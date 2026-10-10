{ (H9) Calling Forward(X) through a derived type whose override omits the
  default: expected to FAIL to compile. }
program xfail_h9_nodefault_derived;
{$mode objfpc}{$H+}
type
  TModule = class
    function Forward(X: Integer; Train: Boolean = False): Integer; virtual; abstract;
  end;
  TDropout = class(TModule)
    function Forward(X: Integer; Train: Boolean): Integer; override;
  end;
function TDropout.Forward(X: Integer; Train: Boolean): Integer;
begin
  Result := X;
end;
var
  D: TDropout;
begin
  D := TDropout.Create;
  WriteLn(D.Forward(1));
  D.Free;
end.
