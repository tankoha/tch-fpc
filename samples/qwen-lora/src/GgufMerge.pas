{ GgufMerge — the D3 writer (T18), per DESIGN.md 8.15 and
  tasks/T18-facts.md (authoritative): a base GGUF and a checkpoint's
  adapters become one GGUF whose adapted matrices hold W + s·B·A in Q8_0;
  the metadata is copied as bytes, every other tensor byte for byte.
  Interface by the harness owner (tasks/T18-facts.md); the bodies are
  written in T18b1 (MergeTargets, LoadAdapters, MergeTensor, WriteMerged)
  and T18b2 (RunMergeCLI). }
unit GgufMerge;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Tch, GgufFile, GgufQuant, GgufWriter, QwenBlocks, Lora,
  Checkpoint;

const
  GgufTypeQ8_0 = 8;

type
  EGgufMerge = class(Exception);

  { One adapted matrix: the adapter name (as TQwenModel.AttachLora gives
    it), the index of '<Name>.weight' in the base's directory, its
    dimensions (GGUF Dims[0] = InDim, Dims[1] = OutDim). }
  TMergeTarget = record
    Name: string;
    Tensor: Integer;
    InDim, OutDim: Int64;
  end;
  TMergeTargets = array of TMergeTarget;

{ The adapted matrices of AGguf: per trunk block 0..trunk-1 its kind's
  LoraTargets (linear: attn_qkv, attn_gate, ssm_out; full: attn_q,
  attn_k, attn_v, attn_output). Raises EGgufMerge when '<name>.weight'
  is missing or not two-dimensional. }
function MergeTargets(AGguf: TGgufFile): TMergeTargets;

{ A TLoraSet of AState's rank and scale with one CPU pair per target,
  loaded from ADir's lora.ot (the caller frees it). }
function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;

{ The tensor of type ATType (AIn × AOut elements in ABytes) dequantised
  to fp32, plus AP.MergedDelta, quantised to Q8_0. Raises EGgufMerge when
  AP's dimensions differ from AIn, AOut. }
function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;

{ Writes AGguf to AOutPath with the pairs of ASet merged (nil: no pair,
  a byte-identical copy). Written to AOutPath + '.part' and renamed; the
  part file is removed on failure. Returns the number of merged tensors. }
function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;

{ qlora_merge [--base-sha256 H] BASE CKPT OUT; AOut / AErr caller-owned;
  the result is the exit code (0 done, 1 error, 2 usage). }
function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

implementation

function MergeTargets(AGguf: TGgufFile): TMergeTargets;
begin
  Result := nil;
  raise EGgufMerge.Create('MergeTargets: not implemented yet');
end;

function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;
begin
  Result := nil;
  raise EGgufMerge.Create('LoadAdapters: not implemented yet');
end;

function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;
begin
  Result := nil;
  raise EGgufMerge.Create('MergeTensor: not implemented yet');
end;

function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;
begin
  Result := 0;
  raise EGgufMerge.Create('WriteMerged: not implemented yet');
end;

function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
begin
  Result := 0;
  raise EGgufMerge.Create('RunMergeCLI: not implemented yet');
end;

end.
