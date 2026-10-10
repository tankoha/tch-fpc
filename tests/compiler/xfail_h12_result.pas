{ (H12) A FUNCTION parameter named "result" is rejected even with
  duplicatelocals (it is the function result symbol, not a member), so the
  generator renames it (result_). Three ops have such a parameter. }
program xfail_h12_result;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
{$modeswitch duplicatelocals}
type
  TT = record
    V: Int64;
    function f(result: Int64): Int64;
  end;
function TT.f(result: Int64): Int64; begin f := result; end;
var
  T: TT;
begin
  WriteLn(T.f(1));
end.
