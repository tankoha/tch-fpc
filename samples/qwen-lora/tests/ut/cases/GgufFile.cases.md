# UT cases — unit `GgufFile` (T01b)

Unit under test: `src/GgufFile.pas` — read-only GGUF reader (`TGgufFile.Open`,
`FindKey`, `FindTensor`, `ReadTensorBytes`, the read-only properties;
`EGgufError`; `GgufTypeName`, `GgufTensorBytes`, `GgufValueTypeName`).
Base: `tasks/T01-facts.md` ("Unit tests") and the checks/errors of
`docs/detailed-design/T01-detailed.md` §2 (IDs in the Notes column).

Test arrangement (facts "Unit tests"): the test program builds small GGUF
files itself with a tiny writer in the test under `{proj}/work/ut/gguf/`
(mktemp-style unique names, little-endian, per the facts' format block).
UT never needs the rehearsal model, GPU or `QLORA_MODEL`. Message text is
only fixed where the facts fix it (`"unsupported tensor type N"`,
detailed design I-1); all other `EGgufError` messages must follow the rule
2.6/11 (name what was wrong and where).

| Case ID | Title | Input (short) | Expected result | Notes |
|---|---|---|---|---|
| UT-GgufFile-001 | Scalars: all integer value types below 64 bit | one key each: uint8, int8, uint16, int16, uint32, int32, representative values incl. negatives | each: correct `VType`, `Count = 1`, exact `IntVal`, no raise | 2.6/4, 2.7; R3 |
| UT-GgufFile-002 | Scalars: uint64 above High(Int64), int64 negative | uint64 `$FFFFFFFFFFFFFFFF`; int64 -5 | stored as is (typecast): uint64 `IntVal` = -1 view, int64 `IntVal` = -5; `VType` correct | 2.6/4, 2.7; §3 |
| UT-GgufFile-003 | Scalars: float32 and float64 | float32 1.5; float64 2.25 (exact binary values) | `FloatVal` = 1.5 / 2.25 as `Double`; `VType` = gvFloat32 / gvFloat64 | 2.6/4, 2.7 |
| UT-GgufFile-004 | Scalars: bool, both 0 and non-zero | bool 0; bool 1 | `IntVal` = 0 / 1; `VType` = gvBool | 2.6/4, 2.7 |
| UT-GgufFile-005 | Scalars: string, incl. empty string and UTF-8 | 'hello'; '' (count 0); 'héllo' as UTF-8 bytes | `StrVal` byte-for-byte equal, never converted; `Count = 1` | 2.6/4, 2.6/9; R4; §5 (empty string) |
| UT-GgufFile-006 | Arrays: every integer element type and bool | arrays of uint8, int8, uint16, int16, uint32, int32, uint64, int64 (2–3 elements each) and an array of bool | each: correct `ArrType`, `Count`, `Ints` in order (bool elements 0/1) | 2.6/4, 2.7; R3 |
| UT-GgufFile-007 | Arrays: float32 and float64 | 2–3-element arrays with exact binary values | correct `ArrType`, `Count`, `Floats` in order | 2.6/4, 2.7; R3 |
| UT-GgufFile-008 | Arrays: strings, incl. empty-string member | 3 string elements, one '' | `ArrType` gvString, `Count = 3`, `Strs` byte-for-byte | 2.6/9, 2.7; R4 |
| UT-GgufFile-009 | Empty array (count 0) | array of uint8, count 0 | `Count = 0`, `Length(Ints) = 0`, no raise | 2.6/4, 2.7; §5 (empty array) |
| UT-GgufFile-010 | Unknown value type (13) | key with value-type UInt32 = 13 | `EGgufError`; message names the key and byte position | 2.6/5, 2.6/11; R3 |
| UT-GgufFile-011 | Arrays of arrays | array whose element type is array (9) | `EGgufError` | 2.6/6; R3 |
| UT-GgufFile-012 | Key with upper-case letters: stored as read, lookup exact | keys 'MODEL' and 'model.x' | `KV` holds 'MODEL' unchanged; `FindKey('MODEL') >= 0`, `FindKey('model') = -1` | 2.6/9, 2.8; R11; §5 (upper-case key) |
| UT-GgufFile-013 | FindKey: absent → -1; duplicate key → first index | key 'k' twice (different values); lookup of 'nope' | `FindKey('nope') = -1`; `FindKey('k')` = first index | 2.8; R11; §5 (duplicate) |
| UT-GgufFile-014 | FindTensor: exact, case-sensitive; absent; duplicate name | tensors 'wt' and 'WT' | `FindTensor('WT')` = its index, `FindTensor('wt')` = first 'wt', `FindTensor('nope') = -1` | 2.9; R11; §5 (duplicate) |
| UT-GgufFile-015 | Bad magic | first 4 bytes 'XXXX' | `EGgufError`; message names position 0 | 2.6/1, 2.6/11; R1 |
| UT-GgufFile-016 | Version 1 and 4 rejected | two valid files with version 1 and 4 | `EGgufError` each; message names the version | 2.6/2, 2.6/11; R1 |
| UT-GgufFile-017 | Version 2 and 3 accepted | two valid files with version 2 and 3 | `Open` succeeds; `Version` = 2 / 3 | 2.6/2; R1 |
| UT-GgufFile-018 | Empty file (0 bytes) | 0-byte file | `EGgufError` (any read past the end) | 2.6/10; R10 |
| UT-GgufFile-019 | File cut in the middle of a key string | key string declares N bytes, only N-3 present | `EGgufError` (read past end) | 2.6/10, 2.6/11; R10; §5 |
| UT-GgufFile-020 | File cut in the header (magic + 9 bytes) | 13-byte file | `EGgufError` | 2.6/3, 2.6/10 |
| UT-GgufFile-021 | general.alignment absent → 32 | file without the key; directory end not a multiple of 32 | `Alignment = 32`; `DataOffset` = position after directory rounded up to 32 | 2.6/7; R2 |
| UT-GgufFile-022 | general.alignment = 64 present → used | uint32 key with 64; layout ends strictly between two multiples of 64 | `Alignment = 64`; `DataOffset` = rounded up to 64 (rounding exercised) | 2.6/7; R2 |
| UT-GgufFile-023 | general.alignment not uint32 (stored as int64) | key with value type int64, value 64 | `EGgufError`; message names the key | 2.6/7, 2.6/11; R2; §5 |
| UT-GgufFile-024 | general.alignment = 0 | uint32 key with 0 | `EGgufError` | 2.6/7; R2; §5 |
| UT-GgufFile-025 | general.alignment not a power of two (48) | uint32 key with 48 | `EGgufError` | 2.6/7; R2; §5 |
| UT-GgufFile-026 | f32 tensor opens, NBytes correct | dims [4, 3] | `NBytes = 48`; `TType = 0`; `Open` OK | 2.4; R5 |
| UT-GgufFile-027 | f16 tensor | dims [6, 4] | `NBytes = 48` (block 1, 2 bytes/element) | 2.4; R5 |
| UT-GgufFile-028 | q8_0 tensor | dims [32] | `NBytes = 34` | 2.4; R5 |
| UT-GgufFile-029 | q4_K tensor | dims [256, 2] | `NBytes = 288` (2 blocks × 144) | 2.4; R5 |
| UT-GgufFile-030 | q5_K tensor | dims [256] | `NBytes = 176` | 2.4; R5 |
| UT-GgufFile-031 | q6_K tensor | dims [256, 3, 2] | `NBytes = 1260` (6 blocks × 210) | 2.4; R5 |
| UT-GgufFile-032 | bf16 tensor | dims [8] | `NBytes = 16` (block 1, 2 bytes/element) | 2.4; R5 |
| UT-GgufFile-033 | Unsupported tensor type (5) | directory entry with type 5 | `EGgufError` "unsupported tensor type 5" | 2.4, 2.6/8; R5; §5 |
| UT-GgufFile-034 | Row length not a multiple of the block size | q8_0 dims [16, 1]; q4_K dims [128] | `EGgufError` each; message names the tensor | 2.4, 2.6/8, 2.6/11; R5 |
| UT-GgufFile-035 | n_dims = 0 rejected | entry with 0 dimensions | `EGgufError` (n_dims must be 1..4) | 2.6/8; R5; §5 |
| UT-GgufFile-036 | n_dims = 5 rejected | entry with 5 dimensions | `EGgufError` | 2.6/8; R5 (facts "Unit tests") |
| UT-GgufFile-037 | Offsets: correct with padding | alignment 32; tensors f32 [4] (16 B) then f32 [8] at offset 32 | `Open` OK; `Offset` = 0 and 32; first offset is 0 | 2.6/8, 2.7; R6, R8; §5 (padding) |
| UT-GgufFile-038 | One tensor out of place | as UT-GgufFile-037 but second offset 17 | `EGgufError`; message names the tensor and position | 2.6/8, 2.6/11; R6; §5 |
| UT-GgufFile-039 | A tensor past the end of the file | f32 [4] declared; file ends 10 bytes after the data section start | `EGgufError` (extent check) | 2.6/8; R7; §5 |
| UT-GgufFile-049 | A tensor ending one byte past the end of the file | f32 [4] at offset 0 (16 B needed); only 15 data bytes present, so the file (79 bytes) ends one byte before TEnd = 64 + 0 + 16 = 80 | `EGgufError` (extent check); the message names the tensor and quotes the real file size 79 (not 0) | 2.6/8, 2.6/11; R7; §5 |
| UT-GgufFile-040 | Zero tensors | n_tensors = 0, one KV entry | `TensorCount = 0`; `DataOffset` = KV end rounded up; no raise | 2.7; R8; §5 (zero tensors) |
| UT-GgufFile-041 | Open with an absolute path | `Open` with the absolute file name | `Path` = the string as given; `FileSize` correct | 2.5, 2.7 |
| UT-GgufFile-042 | Open with a relative path | chdir into the parent dir; `Open` with the bare name | `Open` OK; `Path` = the relative string as given; parsed values identical to the absolute case | 2.5 (facts "Unit tests") |
| UT-GgufFile-043 | KVStart always 24; KVEnd just after the last value | two KV entries of known byte sizes | `KVStart = 24`; `KVEnd = 24` + exact bytes of both entries | 2.7; R9 |
| UT-GgufFile-044 | ReadTensorBytes returns exactly the bytes written | two tensors with distinct byte patterns; each tensor read twice | each read: `Length = NBytes`, bytes equal to the ones written for that tensor (stream repositioning works). (2.10 end-of-file check is unreachable after the 2.6/8 extent check — noted, not tested.) | 2.10, 2.6/8; R12, R10 |
| UT-GgufFile-045 | GgufValueTypeName: all 13 names | gvUInt8 … gvFloat64 (ordinals 0..12) | 'uint8','int8','uint16','int16','uint32','int32','float32','bool','string','array','uint64','int64','float64' | 2.2 |
| UT-GgufFile-046 | GgufTypeName: the 7 known ids and unknown ids | 0,1,8,12,13,14,30; and 5; 999 | f32, f16, q8_0, q4_K, q5_K, q6_K, bf16; 'type5'; 'type999' | 2.3 |
| UT-GgufFile-047 | GgufTensorBytes: valid, unknown type, bad row length | (0,[1]) → 4; (8,[32]) → 34; (5,[4]) → raises; (12,[128]) → raises | exact byte counts for valid; `EGgufError` with the 2.4 messages for both invalid calls | 2.4 |
| UT-GgufFile-048 | f16 tensor with dims [3, 2] (originally standalone) | dims [3, 2] | `NBytes = 12` | [廃止] merged into UT-GgufFile-027 (that case is now 2-dimensional) |
