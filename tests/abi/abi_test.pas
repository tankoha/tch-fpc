{ ABI verification in Pascal (HANDOFF step 4): measure how FPC 3.2.2 sees
  libtch's C ABI instead of assuming it.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -FUbuild/abi -Flbuild/libtch -k-rpath=$PWD/build/libtch \
      tests/abi/abi_test.pas -obuild/abi_test
    build/abi_test          # all checks
    build/abi_test nilarg   # nil tensor to a non-optional argument }
program abi_test;

uses
  SysUtils, ctypes;

const
  LibTch = 'tch';

type
  PTorchTensor = Pointer;
  PPTorchTensor = ^PTorchTensor;

function get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;
procedure at_free(t: PTorchTensor); cdecl; external LibTch;
function at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function at_is_sparse(t: PTorchTensor): cint; cdecl; external LibTch;
function at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
function at_double_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Double; cdecl; external LibTch;

// C99 bool, three ways: Boolean, LongBool (what ctypes' cbool is), and the
// raw 32-bit register so we can see what the callee leaves in EAX.
function at_autocast_is_enabled: Boolean; cdecl; external LibTch;
function at_autocast_set_enabled(b: Boolean): Boolean; cdecl; external LibTch;
function at_autocast_is_enabled_raw: LongWord; cdecl;
  external LibTch name 'at_autocast_is_enabled';
function at_context_has_openmp: Boolean; cdecl; external LibTch;
function at_context_has_cuda: Boolean; cdecl; external LibTch;
function at_context_has_openmp_lb: LongBool; cdecl;
  external LibTch name 'at_context_has_openmp';
function at_context_has_cuda_lb: LongBool; cdecl;
  external LibTch name 'at_context_has_cuda';
function at_context_has_openmp_raw: LongWord; cdecl;
  external LibTch name 'at_context_has_openmp';
function at_context_has_cuda_raw: LongWord; cdecl;
  external LibTch name 'at_context_has_cuda';

procedure atg_empty(out_: PPTorchTensor; size_data: PInt64; size_len: cint;
  options_kind, options_device: cint); cdecl; external LibTch;
procedure atg_ones(out_: PPTorchTensor; size_data: PInt64; size_len: cint;
  options_kind, options_device: cint); cdecl; external LibTch;
procedure atg_add(out_: PPTorchTensor; self, other: PTorchTensor); cdecl;
  external LibTch;
procedure atg_argmax(out_: PPTorchTensor; self: PTorchTensor; dim_v: Int64;
  dim_null: Byte; keepdim: cint); cdecl; external LibTch;
procedure atg_to_sparse(out_: PPTorchTensor; self: PTorchTensor;
  layout: ShortInt; blocksize_data: PInt64; blocksize_len: cint;
  dense_dim_v: Int64; dense_dim_null: Byte); cdecl; external LibTch;

const
  kFloat = 6;
  CpuDevice = -1;
  KindNames: array[0..45] of String = (
    'Byte', 'Char', 'Short', 'Int', 'Long', 'Half', 'Float', 'Double',
    'ComplexHalf', 'ComplexFloat', 'ComplexDouble', 'Bool', 'QInt8',
    'QUInt8', 'QInt32', 'BFloat16', 'QUInt4x2', 'QUInt2x4', 'Bits1x8',
    'Bits2x4', 'Bits4x2', 'Bits8', 'Bits16', 'Float8_e5m2',
    'Float8_e4m3fn', 'Float8_e5m2fnuz', 'Float8_e4m3fnuz', 'UInt16',
    'UInt32', 'UInt64', 'UInt1', 'UInt2', 'UInt3', 'UInt4', 'UInt5',
    'UInt6', 'UInt7', 'Int1', 'Int2', 'Int3', 'Int4', 'Int5', 'Int6',
    'Int7', 'Float8_e8m0fnu', 'Float4_e2m1fn_x2');

var
  Failures: Integer = 0;

procedure Check(Cond: Boolean; const What: String);
begin
  if Cond then
    WriteLn('  ok    ', What)
  else
  begin
    WriteLn('  FAIL  ', What);
    Inc(Failures);
  end;
end;

{ Returns the first line of the pending error (or '') and clears the slot. }
function TakeError: String;
var
  P: PChar;
  I: Integer;
begin
  P := get_and_reset_last_err;
  if P = nil then
    Exit('');
  Result := StrPas(P);
  tchf_free(P);
  I := Pos(#10, Result);
  if I > 0 then
    SetLength(Result, I - 1);
  if Result = '' then
    Result := '(empty message)';
end;

function MakeOnes(const Dims: array of Int64; Kind: cint): PTorchTensor;
begin
  Result := nil;
  atg_ones(@Result, @Dims[0], Length(Dims), Kind, CpuDevice);
end;

procedure TestBool;
var
  Prev: Boolean;
begin
  WriteLn('C99 bool returns and arguments');
  WriteLn(Format('  raw EAX: openmp=$%.8x cuda=$%.8x autocast=$%.8x',
    [at_context_has_openmp_raw, at_context_has_cuda_raw,
     at_autocast_is_enabled_raw]));
  WriteLn('  Boolean:  openmp=', at_context_has_openmp,
    ' cuda=', at_context_has_cuda);
  WriteLn('  LongBool: openmp=', at_context_has_openmp_lb,
    ' cuda=', at_context_has_cuda_lb);
  Check(not at_context_has_cuda, 'has_cuda = False on the CPU build');
  Check(not at_autocast_is_enabled, 'autocast initially disabled');
  Prev := at_autocast_set_enabled(True);
  Check(not Prev, 'set_enabled(True) returns previous False');
  Check(at_autocast_is_enabled, 'autocast enabled after set_enabled(True)');
  Check(at_autocast_is_enabled_raw = 1, 'raw EAX = 1 when enabled');
  Prev := at_autocast_set_enabled(False);
  Check(Prev, 'set_enabled(False) returns previous True');
  Check(not at_autocast_is_enabled, 'autocast disabled again');
  Check(TakeError = '', 'no error pending');
end;

procedure TestKinds;
var
  K, Got: cint;
  T: PTorchTensor;
  Dims: array[0..0] of Int64 = (2);
  Err: String;
begin
  WriteLn('c10::ScalarType values (atg_empty per kind, then at_scalar_type)');
  for K := Low(KindNames) to High(KindNames) do
  begin
    T := nil;
    atg_empty(@T, @Dims[0], 1, K, CpuDevice);
    Err := TakeError;
    if Err <> '' then
      WriteLn(Format('  %2d %-17s cannot create: %s', [K, KindNames[K],
        Copy(Err, 1, 70)]))
    else
    begin
      Got := at_scalar_type(T);
      Check(Got = K, Format('%2d %-17s at_scalar_type = %d', [K,
        KindNames[K], Got]));
      at_free(T);
    end;
  end;
end;

procedure TestOptionalFlags;
var
  T, R: PTorchTensor;
begin
  WriteLn('uint8_t x_null as Byte, int8_t layout as ShortInt');
  T := MakeOnes([2, 3], kFloat);
  R := nil;
  atg_argmax(@R, T, 1, 0, 0);          // dim = 1 -> shape [2]
  Check((TakeError = '') and (at_dim(R) = 1), 'argmax(dim=1): 1-d result');
  at_free(R);
  R := nil;
  atg_argmax(@R, T, 1, 1, 0);          // dim = None -> 0-d
  Check((TakeError = '') and (at_dim(R) = 0), 'argmax(dim=None): 0-d result');
  at_free(R);
  R := nil;
  atg_argmax(@R, T, 7, 0, 0);          // dim = 7 is out of range
  Check(TakeError <> '', 'argmax(dim=7, not null): error');
  if R <> nil then at_free(R);

  R := nil;
  atg_to_sparse(@R, T, -1, nil, 0, 0, 1);   // layout = None (-1)
  Check((TakeError = '') and (at_is_sparse(R) <> 0),
    'to_sparse(layout=-1 -> nullopt): sparse result');
  at_free(R);
  R := nil;
  atg_to_sparse(@R, T, 0, nil, 0, 0, 1);    // layout = Strided (0)
  Check((TakeError = '') and (at_is_sparse(R) = 0),
    'to_sparse(layout=0 Strided): dense result');
  if R <> nil then at_free(R);
  at_free(T);
end;

procedure TestLinkAndDevice;
var
  A, B, S: PTorchTensor;
  Idx: array[0..1] of Int64 = (1, 2);
begin
  WriteLn('Linking and a real op');
  A := MakeOnes([2, 3], kFloat);
  B := MakeOnes([2, 3], kFloat);
  S := nil;
  atg_add(@S, A, B);
  Check(TakeError = '', 'atg_add: no error');
  Check(at_double_value_at_indexes(S, @Idx[0], 2) = 2.0,
    'ones + ones = 2 at [1,2]');
  Check(at_device(S) = -1, 'at_device = -1 (CPU)');
  Check(at_scalar_type(S) = kFloat, 'result kind = Float (6)');
  at_free(S); at_free(B); at_free(A);
end;

procedure TestNilArgument;
var
  A, S: PTorchTensor;
begin
  WriteLn('nil tensor passed to a non-optional argument of atg_add');
  A := MakeOnes([2, 3], kFloat);
  S := nil;
  try
    atg_add(@S, nil, A);
    WriteLn('  returned; error slot: "', TakeError, '"');
  except
    on E: Exception do
      WriteLn('  Pascal exception: ', E.ClassName, ': ', E.Message);
  end;
  WriteLn('  still alive after the call');
end;

begin
  if ParamStr(1) = 'nilarg' then
  begin
    TestNilArgument;
    Exit;
  end;
  TestBool;
  TestKinds;
  TestOptionalFlags;
  TestLinkAndDevice;
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn('FAILED (', Failures, ')');
    Halt(1);
  end;
end.
