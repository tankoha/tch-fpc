{ L2: hand-written external declarations for libtch (csrc/libtch/torch_api.h).
  Types follow DESIGN.md section 2: Boolean (never LongBool) for C bool,
  cint for int, Byte for uint8_t, ShortInt for int8_t, csize_t for size_t. }
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

function get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

procedure at_free(t: PTorchTensor); cdecl; external LibTch;
function at_defined(t: PTorchTensor): cint; cdecl; external LibTch;
function at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl;
  external LibTch;
function at_double_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Double; cdecl; external LibTch;
function at_int64_value_at_indexes(t: PTorchTensor; indexes: PInt64;
  indexes_len: cint): Int64; cdecl; external LibTch;

{ Stand-ins for TchSysGenerated (step 8): the few atg_* the ownership tests
  need, declared exactly as the generator will emit them. }
procedure atg_add(out_: PPTorchTensor; self, other: PTorchTensor); cdecl;
  external LibTch;
procedure atg_mul(out_: PPTorchTensor; self, other: PTorchTensor); cdecl;
  external LibTch;
procedure atg_cat(out_: PPTorchTensor; tensors_data: PPTorchTensor;
  tensors_len: cint; dim: Int64); cdecl; external LibTch;
procedure atg_max_dim(out_: PPTorchTensor; self: PTorchTensor; dim: Int64;
  keepdim: cint); cdecl; external LibTch;
procedure atg_ones(out_: PPTorchTensor; size_data: PInt64; size_len: cint;
  options_kind, options_device: cint); cdecl; external LibTch;

implementation

end.
