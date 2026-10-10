{ (H2) addendum: with both Single and Double overloads, a real literal is
  Single only if its first element is exactly representable in binary32.
  [0.9, -1.9] is not, and the call is ambiguous ("Can't determine which
  overloaded function to call"). Hence no Single overload (DESIGN 4.3). }
program xfail_h2_single;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    class procedure FromArray(const A: array of Single); overload; static;
    class procedure FromArray(const A: array of Double); overload; static;
  end;
class procedure TT.FromArray(const A: array of Single); begin end;
class procedure TT.FromArray(const A: array of Double); begin end;
begin
  TT.FromArray([0.9, -1.9]);
end.
