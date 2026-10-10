{ (H10) A record containing a dynamic array of its own type. }
program h10_ivalue;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
uses SysUtils;
type
  TIValueTag = (ivNone, ivInt, ivDouble, ivString, ivTuple, ivList);
  TIValue = record
    Tag: TIValueTag;
    I: Int64;
    D: Double;
    S: String;
    H: IInterface;               // stands in for a TTensor field
    Items: array of TIValue;
    class function Int(V: Int64): TIValue; static;
    class function Tuple(const A: array of TIValue): TIValue; static;
    function ToString: String;
  end;
class function TIValue.Int(V: Int64): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivInt; Result.I := V;
  Result.H := TInterfacedObject.Create;
end;
class function TIValue.Tuple(const A: array of TIValue): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivTuple;
  SetLength(Result.Items, Length(A));
  for K := 0 to High(A) do
    Result.Items[K] := A[K];
end;
function TIValue.ToString: String;
var
  K: Integer;
begin
  case Tag of
    ivInt: Result := IntToStr(I);
    ivTuple:
      begin
        Result := '(';
        for K := 0 to High(Items) do
        begin
          if K > 0 then Result := Result + ', ';
          Result := Result + Items[K].ToString;
        end;
        Result := Result + ')';
      end;
  else
    Result := '?';
  end;
end;
var
  V, W: TIValue;
begin
  V := TIValue.Tuple([TIValue.Int(1), TIValue.Tuple([TIValue.Int(2),
    TIValue.Int(3)]), TIValue.Int(4)]);
  W := V;                      // copy: shares Items (dynamic array refcount)
  W.Items[0].I := 99;          // copy-on-write? (dynamic arrays are not COW)
  WriteLn('V = ', V.ToString);
  WriteLn('W = ', W.ToString);
end.
