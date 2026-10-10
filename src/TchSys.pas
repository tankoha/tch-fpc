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

function at_tensor_of_data(vs: Pointer; dims: PInt64; ndims: csize_t;
  element_size_in_bytes: csize_t; type_: cint): PTorchTensor; cdecl;
  external LibTch;
procedure at_copy_data(t: PTorchTensor; vs: Pointer; numel: csize_t;
  elt_size_in_bytes: csize_t); cdecl; external LibTch;
function at_shallow_clone(t: PTorchTensor): PTorchTensor; cdecl;
  external LibTch;
procedure at_copy_(dst, src: PTorchTensor); cdecl; external LibTch;

procedure at_backward(t: PTorchTensor; keep_graph, create_graph: cint);
  cdecl; external LibTch;
function at_requires_grad(t: PTorchTensor): cint; cdecl; external LibTch;
function at_grad_set_enabled(b: cint): cint; cdecl; external LibTch;

procedure at_manual_seed(seed: Int64); cdecl; external LibTch;
function at_get_num_threads: cint; cdecl; external LibTch;
procedure at_set_num_threads(n_threads: cint); cdecl; external LibTch;
function atc_cuda_device_count: cint; cdecl; external LibTch;
function atc_cuda_is_available: cint; cdecl; external LibTch;

function ats_int(v: Int64): PTorchScalar; cdecl; external LibTch;
function ats_float(v: Double): PTorchScalar; cdecl; external LibTch;
procedure ats_free(s: PTorchScalar); cdecl; external LibTch;

{ Serialization (DESIGN.md 4.6.3). The callback receives a new tensor it
  must own and a name borrowed for the duration of the call. It must never
  raise: a Pascal exception may not cross the C++ frames. }
type
  TTorchLoadCallback = procedure(data: Pointer; name: PChar; t: PTorchTensor);
    cdecl;

procedure at_save_multi(tensors: PPTorchTensor; tensor_names: PPChar;
  ntensors: cint; filename: PChar); cdecl; external LibTch;
procedure at_load_callback_with_device(filename: PChar; data: Pointer;
  f: TTorchLoadCallback; device_id: cint); cdecl; external LibTch;
procedure at_loadz_callback_with_device(filename: PChar; data: Pointer;
  f: TTorchLoadCallback; device_id: cint); cdecl; external LibTch;

{ Optimizers (DESIGN.md 4.6.4). The constructors return nil on error. }
type
  PTorchOptimizer = Pointer;

function ato_sgd(learning_rate, momentum, dampening, weight_decay: Double;
  nesterov: cint): PTorchOptimizer; cdecl; external LibTch;
function ato_adam(learning_rate, beta1, beta2, weight_decay, eps: Double;
  amsgrad: Boolean): PTorchOptimizer; cdecl; external LibTch;
function ato_adamw(learning_rate, beta1, beta2, weight_decay, eps: Double;
  amsgrad: Boolean): PTorchOptimizer; cdecl; external LibTch;
function ato_rms_prop(learning_rate, alpha, eps, weight_decay,
  momentum: Double; centered: cint): PTorchOptimizer; cdecl; external LibTch;
procedure ato_add_parameters(opt: PTorchOptimizer; t: PTorchTensor;
  group: csize_t); cdecl; external LibTch;
procedure ato_set_learning_rate(opt: PTorchOptimizer;
  learning_rate: Double); cdecl; external LibTch;
procedure ato_set_learning_rate_group(opt: PTorchOptimizer; group: csize_t;
  learning_rate: Double); cdecl; external LibTch;
procedure ato_set_momentum(opt: PTorchOptimizer; momentum: Double); cdecl;
  external LibTch;
procedure ato_set_momentum_group(opt: PTorchOptimizer; group: csize_t;
  momentum: Double); cdecl; external LibTch;
procedure ato_set_weight_decay(opt: PTorchOptimizer;
  weight_decay: Double); cdecl; external LibTch;
procedure ato_set_weight_decay_group(opt: PTorchOptimizer; group: csize_t;
  weight_decay: Double); cdecl; external LibTch;
procedure ato_zero_grad(opt: PTorchOptimizer); cdecl; external LibTch;
procedure ato_step(opt: PTorchOptimizer); cdecl; external LibTch;
procedure ato_free(opt: PTorchOptimizer); cdecl; external LibTch;

implementation

end.
