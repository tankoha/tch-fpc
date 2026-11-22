{ T10b4 UT fixture (for UT-QwenBlocks-216): writes a GGUF v3 file with the
  tiny model's qwen35 metadata of tests/ut/QwenBlocksFixture.pas (same 16
  keys and values) but WITHOUT qwen35.rope.freq_base and with zero tensors,
  so ReadQwenDims fails on the first float key it reads. Covers the
  missing-key raise of KeyFloat in src/QwenBlocks.pas (line coverage; the
  float-key counterpart of the UT-QwenBlocks-102 path through KeyInt).
  The writer mirrors the harness fixture (harness owner's, unchanged).
  Not code under test. }
unit QwenBlocksMetaFixture;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

{ The tiny model's metadata (same keys and values as WriteTinyQwen) except
  qwen35.rope.freq_base; see the unit header. }
procedure WriteQwenMetaNoRopeBase(const APath: string);

implementation

procedure WU32(S: TStream; V: Cardinal); begin S.WriteBuffer(V, 4); end;
procedure WU64(S: TStream; V: QWord); begin S.WriteBuffer(V, 8); end;
procedure WF32(S: TStream; V: Single); begin S.WriteBuffer(V, 4); end;
procedure WStr(S: TStream; const V: RawByteString);
begin
  WU64(S, Length(V));
  if V <> '' then S.WriteBuffer(V[1], Length(V));
end;

procedure KvU32(S: TStream; const K: RawByteString; V: Cardinal);
begin WStr(S, K); WU32(S, 4); WU32(S, V); end;
procedure KvF32(S: TStream; const K: RawByteString; V: Single);
begin WStr(S, K); WU32(S, 6); WF32(S, V); end;
procedure KvStr(S: TStream; const K, V: RawByteString);
begin WStr(S, K); WU32(S, 8); WStr(S, V); end;

procedure WriteQwenMetaNoRopeBase(const APath: string);
const
  Align = 32;
var
  S: TFileStream;
  Pos0: QWord;
  Pad: array[0..31] of Byte;
begin
  FillChar(Pad, SizeOf(Pad), 0);
  S := TFileStream.Create(APath, fmCreate);
  try
    S.WriteBuffer(PAnsiChar('GGUF')^, 4);
    WU32(S, 3);
    WU64(S, 0); { tensor count }
    WU64(S, 15); { key-value count }
    KvStr(S, 'general.architecture', 'qwen35');
    KvU32(S, 'qwen35.block_count', 2);
    KvU32(S, 'qwen35.embedding_length', 64);
    KvU32(S, 'qwen35.feed_forward_length', 96);
    KvU32(S, 'qwen35.attention.head_count', 4);
    KvU32(S, 'qwen35.attention.head_count_kv', 2);
    KvU32(S, 'qwen35.attention.key_length', 32);
    KvU32(S, 'qwen35.rope.dimension_count', 16);
    { qwen35.rope.freq_base is deliberately absent (UT-QwenBlocks-216) }
    KvF32(S, 'qwen35.attention.layer_norm_rms_epsilon', 1e-6);
    KvU32(S, 'qwen35.ssm.group_count', 2);
    KvU32(S, 'qwen35.ssm.time_step_rank', 4);
    KvU32(S, 'qwen35.ssm.state_size', 16);
    KvU32(S, 'qwen35.ssm.conv_kernel', 4);
    KvU32(S, 'qwen35.full_attention_interval', 2);
    KvU32(S, 'qwen35.attention.value_length', 32);
    Pos0 := S.Position;
    if Pos0 mod Align <> 0 then
      S.WriteBuffer(Pad, Align - Pos0 mod Align);
  finally
    S.Free;
  end;
end;

end.
