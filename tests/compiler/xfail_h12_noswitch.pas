{ (H12) Baseline without the switch: in {$mode objfpc} a parameter named
  like a member declared earlier is a duplicate identifier. (Declared later
  it compiles, so the generator never relies on declaration order.) }
program xfail_h12_noswitch;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    V: Int64;
    function Dim: Int64;
    function sum(dim: Int64): Int64;
  end;
function TT.Dim: Int64; begin Result := V; end;
function TT.sum(dim: Int64): Int64; begin Result := V + dim; end;
var
  T: TT;
begin
  T.V := 1;
  WriteLn(T.sum(2));
end.
