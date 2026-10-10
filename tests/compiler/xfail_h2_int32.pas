{ (H2) Expected to FAIL to compile: with overloads on both array of Int32
  and array of Int64, an integer literal open array is ambiguous. }
program xfail_h2_int32;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    Tag: String;
    class function FromArray(const A: array of Int64): TT; overload; static;
    class function FromArray(const A: array of Int32): TT; overload; static;
  end;
class function TT.FromArray(const A: array of Int64): TT; begin Result.Tag := 'Int64'; end;
class function TT.FromArray(const A: array of Int32): TT; begin Result.Tag := 'Int32'; end;
begin
  WriteLn(TT.FromArray([1, 2]).Tag);
end.
