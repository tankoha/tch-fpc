#!/bin/bash
# Generates big_record.pas: a TTensor-shaped record with N methods pulled in
# through {$I} inside the record declaration (as the generator will do), plus
# an indexed default property for T[i]. Usage: gen_big.sh OUTDIR [N]
set -euo pipefail
out=$1
n=${2:-2657}
mkdir -p "$out"
{
  for ((i = 0; i < n; i++)); do
    echo "    function op_$i(const other: TT; dim_: Int64; keepdim: Boolean = False): TT;"
  done
} > "$out/big_intf.inc"
{
  for ((i = 0; i < n; i++)); do
    cat <<PAS
function TT.op_$i(const other: TT; dim_: Int64; keepdim: Boolean): TT;
var
  Outs: array[0..0] of Pointer;
begin
  TchCheckArg(Self, 'self', 'op_$i');
  TchCheckArg(other, 'other', 'op_$i');
  Outs[0] := nil;
  stub_call(@Outs[0], H, other.H, dim_, Ord(keepdim));
  TchCheck;
  Result := TT.FromRaw(Outs[0]);
end;

PAS
  done
} > "$out/big_impl.inc"
cat > "$out/big_record.pas" <<'PAS'
program big_record;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
uses SysUtils;
type
  TT = record
    H: Pointer;
    M: IInterface;               // managed field, as in the real TTensor
    class function FromRaw(P: Pointer): TT; static;
    function GetItem(I: Int64): TT;
    property Items[I: Int64]: TT read GetItem; default;
    {$I big_intf.inc}
  end;
procedure stub_call(o: PPointer; a, b: Pointer; d: Int64; k: Integer); cdecl;
begin
  o^ := Pointer(PtrUInt(a) + PtrUInt(d) + PtrUInt(k));
end;
procedure TchCheck; inline;
begin
end;
procedure TchCheckArg(const T: TT; const P, O: String); inline;
begin
  if T.H = nil then
    raise Exception.Create(P + ' ' + O);
end;
class function TT.FromRaw(P: Pointer): TT;
begin
  Result.H := P;
end;
function TT.GetItem(I: Int64): TT;
begin
  Result.H := Pointer(PtrUInt(H) + PtrUInt(I));
end;
{$I big_impl.inc}
var
  T: TT;
begin
  T.H := Pointer(1);
  WriteLn('op_0(T, 2)          -> ', PtrUInt(T.op_0(T, 2).H));
  WriteLn('op_2656(T, 3, True) -> ', PtrUInt(T.op_2656(T, 3, True).H));
  WriteLn('T[5]                -> ', PtrUInt(T[5].H));
  WriteLn('T[5][2]             -> ', PtrUInt(T[5][2].H));
end.
PAS
