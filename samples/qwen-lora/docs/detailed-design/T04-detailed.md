# TGgufWriter Detailed Design (T04)

---

## 1. Create (constructor)

*Step 1 – 3 (T04‑facts.md)*
1. Open the source `TGgufFile` (`ASrc`) and read the header:
   * version, alignment, `KVCount`, and the positions `KVStart..KVEnd`.
2. Copy the metadata bytes verbatim into the writer’s internal buffer.
3. Build an internal array of `TGgufTensorInfo` from the source’s tensor directory, preserving the original order, names, `NDims`, and dimension values.
4. Call the internal **Layout** routine (see §2) to compute for every tensor:
   * `NBytes` via `GgufTensorBytes` with the current `TType`.
   * `Offset` (cumulative, padded to the writer’s alignment).
5. Set `DataOffset` and `FileSize` based on the summed offsets and padded sizes.
6. Store `TensorCount` = source’s tensor count.

*Errors* – none; `Create` never raises – it only fails if the source cannot be opened, which is handled by `TGgufFile` itself.

---

## 2. Destroy (destructor)

*Step 1 (T04‑facts.md – row 114‑115)*
1. If `BeginWrite` succeeded **and** `Finish` has **not** been called, delete the partially‑written output file so no half‑written file remains.
2. Release the internal buffers (metadata, tensor array).

---

## 3. SetTensorType

*Steps 1‑4 (T04‑facts.md – rows 68‑70, 102‑103)*
1. Verify that the writer is **not** in a `BeginWrite` phase; otherwise raise `EGgufError` (“SetTensorType after BeginWrite”).
2. Locate the tensor by exact, case‑sensitive name `AName`. If not found, raise `EGgufError` with message `gguf writer: no tensor named "<name>"` (rows 007‑008).
3. Call `GgufTensorBytes` **before** mutating the tensor to ensure the new type is supported and the row length matches the block size of the type. Propagate any `EGgufError` (unknown type, row‑length mismatch – rows 009‑010).
4. On success, replace the tensor’s `TType` with `ATType` and **re‑run** `Layout` to recompute all offsets, `NBytes`, `DataOffset` and `FileSize`.

---

## 4. BeginWrite

*Steps 1‑6 (T04‑facts.md – rows 73‑75, 106‑107)*
1. Ensure the writer has not already begun writing; otherwise raise `EGgufError` (row 106).
2. Attempt to create the output file `APath` with `TFileStream.Create(APath, fmCreate)`. If the directory does not exist or the file cannot be created, translate the `EStreamError` into `EGgufError` with a message containing the path (row 107).
3. Write the 24‑byte header (magic, version, tensor count, metadata key count).
4. Write the copied metadata bytes (`KVStart..KVEnd`).
5. Write the tensor directory entries using the layout computed by **Layout**. Each entry consists of:
   * name length, name bytes, `NDims`, the `NDims` dimension values, `TType`, `Offset`.
6. Pad with zero bytes up to the next multiple of the writer’s alignment; this position becomes `DataOffset`.
7. At this point the file contains header + metadata + directory + alignment padding; no tensor data is written yet.

---

## 5. WriteTensor

*Steps 1‑4 (T04‑facts.md – rows 78‑81, 108‑110, 111‑112)*
1. Verify that `BeginWrite` has been called; otherwise raise `EGgufError` (row 108).
2. Determine the next tensor to write from the internal directory order. If all tensors are already written, raise `EGgufError` (row 110) with a message like `already wrote N of M tensors`.
3. Check that `Length(ABytes)` equals the tensor’s planned `NBytes`. If not, raise `EGgufError` with a message containing the tensor name, the expected byte count, and the actual length (row 109).
4. Write `ABytes` to the file at the pre‑computed `Offset`.
5. Write zero padding up to the next alignment boundary (relative to `DataOffset`).
6. Increment the internal “written count”.

---

## 6. Finish

*Steps 1‑3 (T04‑facts.md – rows 82‑84, 111‑113)*
1. Verify that `BeginWrite` has been called; otherwise raise `EGgufError` (row 111).
2. Verify that **all** tensors have been written (`writtenCount = TensorCount`). If not, raise `EGgufError` with a message `N of M tensors written` (row 112).
3. Close the output file handle; the file now contains a complete, valid GGUF image.

---

## 7. GgufCopy (utility procedure)

*Steps 1‑3 (T04‑facts.md – rows 91‑95, 173‑176)*
1. Open the source file via `TGgufFile.Open(ASrcPath)`.
2. Create a temporary `TGgufWriter` with the source, call `BeginWrite(ADstPath)`, then sequentially read each tensor’s raw bytes from the source (`ReadTensorBytes`) and feed them to `WriteTensor`.
3. Call `Finish` and free the writer. The result is a byte‑identical copy of the source when no tensor types were changed (identity property, rows 91‑94).

---

## 8. Layout (internal routine)

*Steps 1‑5 (T04‑facts.md – implementation notes, rows 98‑101)*
1. For each tensor in the internal array, compute `NBytes = GgufTensorBytes(TType, Slice(T.Dims, T.NDims))`.
2. Starting from offset **0**, compute each tensor’s `Offset` as the previous tensor’s `Offset + AlignUp(previous NBytes)` where `AlignUp(x) = ((x + A - 1) div A) * A` and **A** is the writer’s alignment (taken from the source’s `general.alignment` or default 32).
3. After the last tensor, compute `DataOffset = 24 + (KVEnd‑KVStart) + Σ(entry sizes)`, then pad to the next multiple of **A**.
4. `FileSize` = `DataOffset + Σ( AlignUp(NBytes_i) )`.
5. Store the computed values in the writer’s public properties (`Tensor[i].Offset`, `Tensor[i].NBytes`, `DataOffset`, `FileSize`).

---

## 9. Error Conditions (each listed with the corresponding rows in T04‑facts.md)

| Error | When | Fact rows |
|---|---|---|
| **No tensor named "x"** | `SetTensorType` with a name not present in the source directory. | 007, 108 |
| **Upper‑case name** | Same as above; the name lookup is case‑sensitive. | 008 |
| **Row length not a multiple of block size** | `SetTensorType` attempts to assign a type whose block size does not divide the tensor’s row length (e.g., `b → Q8_0`). | 009 |
| **Unsupported tensor type** | `SetTensorType` receives an unknown type code (e.g., 99). | 010 |
| **BeginWrite called twice** | Subsequent call to `BeginWrite` after a successful first call. | 106 |
| **BeginWrite to non‑existent directory** | File creation fails because the target directory does not exist. | 107 |
| **WriteTensor before BeginWrite** | `WriteTensor` invoked without a preceding `BeginWrite`. | 108 |
| **WriteTensor length mismatch** | Supplied byte array length does not equal the planned `NBytes` for the tensor. | 109 |
| **WriteTensor after all tensors written** | Attempt to write more tensors than present in the directory. | 110 |
| **Finish before BeginWrite** | `Finish` called without a prior `BeginWrite`. | 111 |
| **Finish with missing tensors** | Fewer tensors written than planned (`N of M tensors written`). | 112 |
| **SetTensorType after BeginWrite** | Mutating the layout after the file has been opened for writing. | 113 |
| **Free after BeginWrite without Finish** | Destructor runs while a partial file exists; the file is deleted. | 114 |
| **GgufCopy of missing source** | `TGgufFile.Open` fails (e.g., source file not found) and the exception propagates (`EFOpenError`). | 115 |

---

*All step numbers refer to the row identifiers in `tasks/T04-facts.md`.*
