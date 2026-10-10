{ L2: hand-written external declarations for libtch (csrc/libtch/torch_api.h).
  Types follow DESIGN.md section 2: Boolean (never LongBool) for C bool,
  cint for int, Byte for uint8_t, ShortInt for int8_t, csize_t for size_t.
  The generated atg_* declarations are in TchSysGenerated.pas. }
unit TchSys;

{$mode objfpc}{$H+}

interface

uses
  ctypes;

const
  LibTch = 'tch';

type
  PTorchTensor = Pointer;
  PPTorchTensor = ^PTorchTensor;
  PTorchScalar = Pointer;

function get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

procedure at_free(t: PTorchTensor); cdecl; external LibTch;
function at_defined(t: PTorchTensor): cint; cdecl; external LibTch;
function at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl;
  external LibTch;
function at_double_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Double; cdecl; external LibTch;
function at_int64_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Int64; cdecl; external LibTch;

function ats_int(v: Int64): PTorchScalar; cdecl; external LibTch;
function ats_float(v: Double): PTorchScalar; cdecl; external LibTch;
procedure ats_free(s: PTorchScalar); cdecl; external LibTch;

implementation

end.
