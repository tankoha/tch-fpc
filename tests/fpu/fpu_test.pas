{ FPU exception mask tests (HANDOFF step 5, H4/H5). Each mode runs in its
  own process because the unmasked cases are expected to crash.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -FUbuild/fpu -Flbuild/libtch -k-rpath=$PWD/build/libtch \
      tests/fpu/fpu_test.pas -obuild/fpu_test
    tests/fpu/run.sh

  Modes:
    nomask-small  FPC default MXCSR, 1/0 on a 4-element tensor
    nomask-large  FPC default MXCSR, 1/0 on 10^7 elements (intra-op pool)
    mask          SetExceptionMask(all) first, then both sizes
    pool-first    create libtorch's pool unmasked, then mask, then 1/0 large
    thread        (H4) mask in main, then 1/0 inside a TThread
    intdiv        (H5) mask, catch an integer EDivByZero, then 1/0 again }
program fpu_test;

uses
  {$ifdef unix}cthreads,{$endif} Classes, SysUtils, Math, ctypes;

const
  LibTch = 'tch';
  kFloat = 6;
  CpuDevice = -1;

type
  PTorchTensor = Pointer;
  PPTorchTensor = ^PTorchTensor;

function get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;
procedure at_free(t: PTorchTensor); cdecl; external LibTch;
function at_get_num_threads: cint; cdecl; external LibTch;
function at_double_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Double; cdecl; external LibTch;
procedure atg_ones(out_: PPTorchTensor; size_data: PInt64; size_len: cint;
  options_kind, options_device: cint); cdecl; external LibTch;
procedure atg_zeros(out_: PPTorchTensor; size_data: PInt64; size_len: cint;
  options_kind, options_device: cint); cdecl; external LibTch;
procedure atg_div(out_: PPTorchTensor; self, other: PTorchTensor); cdecl;
  external LibTch;
procedure atg_add(out_: PPTorchTensor; self, other: PTorchTensor); cdecl;
  external LibTch;

function ReadMXCSR: LongWord; assembler; nostackframe;
asm
  sub    $8, %rsp
  stmxcsr (%rsp)
  movl   (%rsp), %eax
  add    $8, %rsp
end;

function MaskStr: String;
begin
  Result := Format('MXCSR=$%.4x x87CW=$%.4x', [ReadMXCSR, Get8087CW]);
end;

procedure TakeError(const What: String);
var
  P: PChar;
begin
  P := get_and_reset_last_err;
  if P <> nil then
  begin
    WriteLn('  ', What, ': libtch error: ', Copy(StrPas(P), 1, 100));
    tchf_free(P);
  end;
end;

function Fill(N: Int64; One: Boolean): PTorchTensor;
begin
  Result := nil;
  if One then
    atg_ones(@Result, @N, 1, kFloat, CpuDevice)
  else
    atg_zeros(@Result, @N, 1, kFloat, CpuDevice);
  TakeError('fill');
end;

{ Computes ones(N) / zeros(N) and reports element 0. }
procedure DivByZero(const Tag: String; N: Int64);
var
  A, B, Q: PTorchTensor;
  Idx: Int64 = 0;
  V: Double;
begin
  A := Fill(N, True);
  B := Fill(N, False);
  Q := nil;
  WriteLn('  ', Tag, ': ones(', N, ') / zeros(', N, ') with ', MaskStr);
  Flush(Output);
  try
    atg_div(@Q, A, B);
    TakeError('div');
    V := at_double_value_at_indexes(Q, @Idx, 1);
    WriteLn('  ', Tag, ': ok, q[0] = ', FloatToStr(V));
    at_free(Q);
  except
    on E: Exception do
      WriteLn('  ', Tag, ': Pascal exception ', E.ClassName, ': ', E.Message);
  end;
  at_free(A); at_free(B);
end;

procedure WarmPool;
var
  A, B, S: PTorchTensor;
begin
  A := Fill(10000000, True);
  B := Fill(10000000, True);
  S := nil;
  atg_add(@S, A, B);
  TakeError('add');
  at_free(S); at_free(A); at_free(B);
end;

procedure MaskAll;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
end;

type
  TDivThread = class(TThread)
  protected
    procedure Execute; override;
  end;

procedure TDivThread.Execute;
begin
  WriteLn('  thread start: ', MaskStr);
  DivByZero('thread small', 4);
  DivByZero('thread large', 10000000);
end;

procedure RunThread;
var
  T: TDivThread;
begin
  T := TDivThread.Create(False);
  T.WaitFor;
  T.Free;
end;

procedure IntDiv;
var
  X, Y: Integer;
begin
  Y := 0;
  X := 1;
  try
    X := X div Y;
    WriteLn('  int div: no exception, X = ', X);
  except
    on E: Exception do
      WriteLn('  int div: caught ', E.ClassName, ': ', E.Message);
  end;
end;

var
  Mode: String;
begin
  Mode := ParamStr(1);
  WriteLn('mode ', Mode, ': at start ', MaskStr, ', torch threads ',
    at_get_num_threads);
  if Mode = 'nomask-small' then
    DivByZero('main small', 4)
  else if Mode = 'nomask-large' then
    DivByZero('main large', 10000000)
  else if Mode = 'mask' then
  begin
    MaskAll;
    DivByZero('main small', 4);
    DivByZero('main large', 10000000);
  end
  else if Mode = 'pool-first' then
  begin
    WarmPool;
    MaskAll;
    DivByZero('main large', 10000000);
  end
  else if Mode = 'thread' then
  begin
    MaskAll;
    RunThread;
  end
  else if Mode = 'intdiv' then
  begin
    MaskAll;
    IntDiv;
    WriteLn('  after int div: ', MaskStr);
    DivByZero('main small', 4);
    DivByZero('main large', 10000000);
  end
  else
  begin
    WriteLn('unknown mode');
    Halt(2);
  end;
  WriteLn('mode ', Mode, ': finished');
end.
