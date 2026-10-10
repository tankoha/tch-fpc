{ (H12) duplicatelocals lifts the check for parameters only: a LOCAL
  VARIABLE in a method body still may not share a member's name in
  {$mode objfpc} (only {$mode delphi} allows that). Hand-written bodies
  therefore cannot shadow a member by accident through a local. }
program xfail_h12_localvar;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
{$modeswitch duplicatelocals}
type
  TT = record
    V: Int64;
    function Dim: Int64;
    function g: Int64;
  end;
function TT.Dim: Int64; begin Result := V; end;
function TT.g: Int64;
var
  dim: Int64;
begin
  dim := 5;
  Result := dim;
end;
var
  T: TT;
begin
  T.V := 7;
  WriteLn(T.g);
end.
