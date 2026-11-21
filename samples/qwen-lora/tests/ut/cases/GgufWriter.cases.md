# UT cases — unit `GgufWriter` (T04)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-GgufWriter-001 | Create, align-32 source | Create | align-32 source | 3 tensors a, w, b; NDims 1, 2, 2; Dims [3], [256,2], [5,1]; types 0, 12, 0; offsets 0, 32, 320; bytes 12, 288, 20; DataOffset 192; FileSize 544 | |
| UT-GgufWriter-002 | Create, align-64 source | Create | align-64 source | offsets 0, 64, 384; DataOffset 256; FileSize 704 | |
| UT-GgufWriter-003 | Create, source without tensors | Create | source without tensors | TensorCount 0; DataOffset 96; FileSize 96 | |
| UT-GgufWriter-004 | w → Q8_0 | SetTensorType | w → Q8_0 | w type 8, offset 32, 544 bytes; b offset 576; a unchanged; FileSize 800 | |
| UT-GgufWriter-005 | w → Q8_0, then w → q4_K | SetTensorType | w → Q8_0, then w → q4_K | the plan of row 001 again | |
| UT-GgufWriter-006 | a → f16 | SetTensorType | a → f16 | a 6 bytes; w 32, b 320; FileSize 544 | |
| UT-GgufWriter-007 | unknown name `x` | SetTensorType | x | EGgufError, message contains `x` | |
| UT-GgufWriter-008 | name `W` (upper case) | SetTensorType | W | EGgufError | |
| UT-GgufWriter-009 | b → Q8_0 | SetTensorType | b → Q8_0 | EGgufError (row length 5); the plan is that of row 001 afterwards | |
| UT-GgufWriter-010 | w → type 99 | SetTensorType | w → type 99 | EGgufError (unsupported type 99); the plan is that of row 001 afterwards | |
| UT-GgufWriter-101 | GgufCopy, align-32 source | GgufCopy | align-32 source | output bytes = source bytes (544) | |
| UT-GgufWriter-102 | GgufCopy, align-64 source | GgufCopy | align-64 source | output bytes = source bytes (704) | |
| UT-GgufWriter-103 | GgufCopy, source without tensors | GgufCopy | source without tensors | output bytes = source bytes (96) | |
| UT-GgufWriter-104 | GgufCopy to a bare relative name | GgufCopy | bare relative name | after `SetCurrentDir(Dir)`: `GgufCopy('src.gguf', 'copy.gguf')`, bytes equal; previous directory restored | |
| UT-GgufWriter-105 | w → Q8_0 written and read back | BeginWrite/WriteTensor/Finish | w → Q8_0 | size 800; 3 tensors; KVCount 1; key `general.architecture` = "test"; DataOffset 192; w type 8 offset 32; b offset 576; ReadTensorBytes of a, w, b = the bytes written; bytes DataOffset+12 .. +31 are 0 | |
| UT-GgufWriter-106 | BeginWrite twice | BeginWrite | (none) | EGgufError | |
| UT-GgufWriter-107 | BeginWrite to `/nonexistent-dir/x.gguf` | BeginWrite | `/nonexistent-dir/x.gguf` | EGgufError, message contains the path | |
| UT-GgufWriter-108 | WriteTensor before BeginWrite | WriteTensor | (none) | EGgufError | |
| UT-GgufWriter-109 | WriteTensor with 11 bytes for `a` | WriteTensor | 11 bytes for a | EGgufError, message contains `a`, 12 and 11 | |
| UT-GgufWriter-110 | WriteTensor after all three | WriteTensor | (none) | EGgufError | |
| UT-GgufWriter-111 | Finish before BeginWrite | Finish | (none) | EGgufError | |
| UT-GgufWriter-112 | Finish after one tensor | Finish | (none) | EGgufError, message contains "1 of 3" | |
| UT-GgufWriter-113 | SetTensorType after BeginWrite | SetTensorType | (none) | EGgufError | |
| UT-GgufWriter-114 | Free after BeginWrite without Finish | Destroy | (none) | the output file does not exist | |
| UT-GgufWriter-115 | GgufCopy of a missing source | GgufCopy | missing source | the exception of `TGgufFile.Open` (measured: EFOpenError) passes through; no destination file | |
| UT-GgufWriter-116 | Finish twice; WriteTensor after Finish | Finish/WriteTensor | all three tensors written, Finish called | second Finish: EGgufError; WriteTensor after Finish: EGgufError | added in T04b3 |
