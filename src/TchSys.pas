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

{ TorchScript modules and ivalues (DESIGN.md 4.6.8). All are PROTECTed
  except atm_free, ati_free and at_set_graph_executor_optimize; the
  constructors return nil on error. atm_named_parameters calls back with
  the load callback's signature. ati_to_string returns a strdup'd string
  (tchf_free). Not declared: atm_fuser_cuda_*. }
type
  PTorchModule = Pointer;
  PTorchIValue = Pointer;
  PPTorchIValue = ^PTorchIValue;

function atm_load(filename: PChar): PTorchModule; cdecl; external LibTch;
function atm_load_on_device(filename: PChar; device: cint): PTorchModule;
  cdecl; external LibTch;
function atm_load_str(data: PChar; sz: csize_t): PTorchModule; cdecl;
  external LibTch;
function atm_load_str_on_device(data: PChar; sz: csize_t;
  device: cint): PTorchModule; cdecl; external LibTch;
function atm_forward(m: PTorchModule; tensors: PPTorchTensor;
  ntensors: cint): PTorchTensor; cdecl; external LibTch;
function atm_forward_(m: PTorchModule; ivalues: PPTorchIValue;
  nivalues: cint): PTorchIValue; cdecl; external LibTch;
function atm_method(m: PTorchModule; method_name: PChar;
  tensors: PPTorchTensor; ntensors: cint): PTorchTensor; cdecl;
  external LibTch;
function atm_method_(m: PTorchModule; method_name: PChar;
  ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl;
  external LibTch;
procedure atm_eval(m: PTorchModule); cdecl; external LibTch;
procedure atm_train(m: PTorchModule); cdecl; external LibTch;
procedure atm_free(m: PTorchModule); cdecl; external LibTch;
procedure atm_to(m: PTorchModule; device, dtype: cint;
  non_blocking: Boolean); cdecl; external LibTch;
procedure atm_save(m: PTorchModule; filename: PChar); cdecl; external LibTch;
function atm_get_profiling_mode: cint; cdecl; external LibTch;
procedure atm_set_profiling_mode(b: cint); cdecl; external LibTch;
procedure atm_set_tensor_expr_fuser_enabled(b: cint); cdecl; external LibTch;
function atm_get_tensor_expr_fuser_enabled: Boolean; cdecl; external LibTch;
procedure at_set_graph_executor_optimize(b: Boolean); cdecl; external LibTch;
procedure atm_named_parameters(m: PTorchModule; data: Pointer;
  f: TTorchLoadCallback); cdecl; external LibTch;
function atm_create_for_tracing(modl_name: PChar; inputs: PPTorchTensor;
  ninputs: cint): PTorchModule; cdecl; external LibTch;
procedure atm_end_tracing(m: PTorchModule; fn_name: PChar;
  outputs: PPTorchTensor; noutputs: cint); cdecl; external LibTch;
{ fork additions: a scripted module from TorchScript source }
function atm_create(modl_name: PChar): PTorchModule; cdecl; external LibTch;
procedure atm_register_parameter(m: PTorchModule; name: PChar;
  t: PTorchTensor; is_buffer: Boolean); cdecl; external LibTch;
procedure atm_define(m: PTorchModule; src: PChar); cdecl; external LibTch;

function ati_none: PTorchIValue; cdecl; external LibTch;
function ati_tensor(t: PTorchTensor): PTorchIValue; cdecl; external LibTch;
function ati_int(v: Int64): PTorchIValue; cdecl; external LibTch;
function ati_double(v: Double): PTorchIValue; cdecl; external LibTch;
function ati_bool(v: cint): PTorchIValue; cdecl; external LibTch;
function ati_string(s: PChar): PTorchIValue; cdecl; external LibTch;
function ati_tuple(ivalues: PPTorchIValue; n: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_generic_list(ivalues: PPTorchIValue; n: cint): PTorchIValue;
  cdecl; external LibTch;
{ 2*npairs ivalues: k0, v0, k1, v1, ... }
function ati_generic_dict(ivalues: PPTorchIValue;
  npairs: cint): PTorchIValue; cdecl; external LibTch;
function ati_int_list(v: PInt64; n: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_double_list(v: PDouble; n: cint): PTorchIValue; cdecl;
  external LibTch;
{ bytes 0/1 }
function ati_bool_list(v: PChar; n: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_string_list(v: PPChar; n: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_tensor_list(v: PPTorchTensor; n: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_device(device: cint): PTorchIValue; cdecl; external LibTch;
function ati_tag(i: PTorchIValue): cint; cdecl; external LibTch;
function ati_to_tensor(i: PTorchIValue): PTorchTensor; cdecl;
  external LibTch;
function ati_to_int(i: PTorchIValue): Int64; cdecl; external LibTch;
function ati_to_double(i: PTorchIValue): Double; cdecl; external LibTch;
function ati_to_string(i: PTorchIValue): PChar; cdecl; external LibTch;
function ati_to_bool(i: PTorchIValue): cint; cdecl; external LibTch;
function ati_length(i: PTorchIValue): cint; cdecl; external LibTch;
function ati_tuple_length(i: PTorchIValue): cint; cdecl; external LibTch;
procedure ati_to_tuple(i: PTorchIValue; outputs: PPTorchIValue; n: cint);
  cdecl; external LibTch;
procedure ati_to_generic_list(i: PTorchIValue; outputs: PPTorchIValue;
  n: cint); cdecl; external LibTch;
{ writes 2*npairs }
procedure ati_to_generic_dict(i: PTorchIValue; outputs: PPTorchIValue;
  npairs: cint); cdecl; external LibTch;
procedure ati_to_int_list(i: PTorchIValue; outputs: PInt64; n: cint);
  cdecl; external LibTch;
procedure ati_to_double_list(i: PTorchIValue; outputs: PDouble; n: cint);
  cdecl; external LibTch;
procedure ati_to_bool_list(i: PTorchIValue; outputs: PChar; n: cint);
  cdecl; external LibTch;
procedure ati_to_tensor_list(i: PTorchIValue; outputs: PPTorchTensor;
  n: cint); cdecl; external LibTch;
procedure ati_free(i: PTorchIValue); cdecl; external LibTch;
function atm_create_class_(m: PTorchModule; clz_name: PChar;
  ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_object_method_(i: PTorchIValue; method_name: PChar;
  ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl;
  external LibTch;
function ati_object_getattr_(i: PTorchIValue; attr_name: PChar):
  PTorchIValue; cdecl; external LibTch;
function ati_clone(i: PTorchIValue): PTorchIValue; cdecl; external LibTch;

implementation

end.
