{ (H2) Overloads of FromArray on the element type: which one do literal
  open arrays pick? Set without Int32: with both Int32 and Int64, [1, 2]
  is ambiguous (see xfail_h2_int32.pas). }
program h2_fromarray;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TT = record
    Tag: String;
    class function FromArray(const A: array of Single): TT; overload; static;
    class function FromArray(const A: array of Double): TT; overload; static;
    class function FromArray(const A: array of Int64): TT; overload; static;
    class function FromArray(const A: array of Byte): TT; overload; static;
    class function FromArray(const A: array of Boolean): TT; overload; static;
  end;
class function TT.FromArray(const A: array of Single): TT; begin Result.Tag := 'Single'; end;
class function TT.FromArray(const A: array of Double): TT; begin Result.Tag := 'Double'; end;
class function TT.FromArray(const A: array of Int64): TT; begin Result.Tag := 'Int64'; end;
class function TT.FromArray(const A: array of Byte): TT; begin Result.Tag := 'Byte'; end;
class function TT.FromArray(const A: array of Boolean): TT; begin Result.Tag := 'Boolean'; end;
var
  S: array of Single = nil;
  D: array of Double = nil;
  I64: array of Int64 = nil;
  X: Double = 1.5;
  N: Integer = 7;
begin
  WriteLn('[1.0, 2.0]       -> ', TT.FromArray([1.0, 2.0]).Tag);
  WriteLn('[1, 2]           -> ', TT.FromArray([1, 2]).Tag);
  WriteLn('[1, 5000000000]  -> ', TT.FromArray([1, 5000000000]).Tag);
  WriteLn('[1.5, 2]         -> ', TT.FromArray([1.5, 2]).Tag);
  WriteLn('[True, False]    -> ', TT.FromArray([True, False]).Tag);
  WriteLn('[X, 2.0] (Double var) -> ', TT.FromArray([X, 2.0]).Tag);
  WriteLn('[N, 2] (Integer var)  -> ', TT.FromArray([N, 2]).Tag);
  SetLength(S, 2); SetLength(D, 2); SetLength(I64, 2);
  WriteLn('array of Single  -> ', TT.FromArray(S).Tag);
  WriteLn('array of Double  -> ', TT.FromArray(D).Tag);
  WriteLn('array of Int64   -> ', TT.FromArray(I64).Tag);
end.
