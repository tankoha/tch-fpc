{ (H2) Overloads of FromArray on the element type: which one do literal
  open arrays pick? The set of cloud session 5 (DESIGN 4.3): no Single
  overload (real literals would be ambiguous or silently rounded, see
  xfail_h2_single.pas) and no Int32 overload ([1, 2] would be ambiguous,
  see xfail_h2_int32.pas); a trailing kind with the natural default.
  A literal array is typed by its first element. }
program h2_fromarray;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  TKind = (kUInt8, kInt64, kFloat, kDouble, kBool);
  TT = record
    Tag: String;
    class function FromArray(const A: array of Double; AKind: TKind = kFloat): TT; overload; static;
    class function FromArray(const A: array of Int64; AKind: TKind = kInt64): TT; overload; static;
    class function FromArray(const A: array of Byte; AKind: TKind = kUInt8): TT; overload; static;
    class function FromArray(const A: array of Boolean; AKind: TKind = kBool): TT; overload; static;
  end;
const
  KindName: array[TKind] of String = ('kUInt8', 'kInt64', 'kFloat', 'kDouble', 'kBool');
class function TT.FromArray(const A: array of Double; AKind: TKind): TT; begin Result.Tag := 'Double ' + KindName[AKind]; end;
class function TT.FromArray(const A: array of Int64; AKind: TKind): TT; begin Result.Tag := 'Int64 ' + KindName[AKind]; end;
class function TT.FromArray(const A: array of Byte; AKind: TKind): TT; begin Result.Tag := 'Byte ' + KindName[AKind]; end;
class function TT.FromArray(const A: array of Boolean; AKind: TKind): TT; begin Result.Tag := 'Boolean ' + KindName[AKind]; end;
var
  D: array of Double = nil;
  I64: array of Int64 = nil;
  X: Double = 1.5;
  S: Single = 1.5;
  N: Integer = 7;
begin
  WriteLn('[1.0, 2.0]       -> ', TT.FromArray([1.0, 2.0]).Tag);
  WriteLn('[0.9, -1.9]      -> ', TT.FromArray([0.9, -1.9]).Tag);
  WriteLn('[1.0, 0.1]       -> ', TT.FromArray([1.0, 0.1]).Tag);
  WriteLn('[3.14]           -> ', TT.FromArray([3.14]).Tag);
  WriteLn('[1.5, 2]         -> ', TT.FromArray([1.5, 2]).Tag);
  WriteLn('[X, 1.5] (Double var) -> ', TT.FromArray([X, 1.5]).Tag);
  WriteLn('[1.5, X] (Double var) -> ', TT.FromArray([1.5, X]).Tag);
  WriteLn('[S] (Single var)      -> ', TT.FromArray([S]).Tag);
  WriteLn('[1, 2]           -> ', TT.FromArray([1, 2]).Tag);
  WriteLn('[1, 5000000000]  -> ', TT.FromArray([1, 5000000000]).Tag);
  WriteLn('[N, 2] (Integer var)  -> ', TT.FromArray([N, 2]).Tag);
  WriteLn('[Byte(200), 1]   -> ', TT.FromArray([Byte(200), 1]).Tag);
  WriteLn('[True, False]    -> ', TT.FromArray([True, False]).Tag);
  WriteLn('[0.9, -1.9], kDouble -> ', TT.FromArray([0.9, -1.9], kDouble).Tag);
  WriteLn('[1, 2], kFloat   -> ', TT.FromArray([1, 2], kFloat).Tag);
  SetLength(D, 2); SetLength(I64, 2);
  WriteLn('array of Double  -> ', TT.FromArray(D).Tag);
  WriteLn('array of Int64   -> ', TT.FromArray(I64).Tag);
end.
