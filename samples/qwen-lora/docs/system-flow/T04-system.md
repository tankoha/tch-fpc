# T04: GgufWriter (system flow)

## Callers

- **T18 (`qlora_merge`):** Writes the merged model. Opens the source GGUF with `GgufFile` (T01), creates a `TGgufWriter` from it, retypes tensors as needed (merged tensors stored as Q8_0), writes each merged tensor's data, and finishes the output file.
- **UT (`tests/ut/GgufWriter_test.pas`):** Reads the output back with T01 `GgufFile` to verify correct layout, metadata, and tensor data.
- **ITa driver (`tests/ita/T04/writer_check.pas`):** Compares the written file with upstream ggml's own GGUF code (`gguf_set_tensor_type`, `gguf_get_meta_data`) and the C reference Q8_0 quantisation.

## Flow

1. `GgufFile` (T01) reads the source GGUF file (header, metadata, tensor directory).
2. `TGgufWriter.Create` plans the output layout:
   - Copies metadata bytes from source
   - Copies tensor entries from source
   - Computes offsets and data offset
3. Optional `SetTensorType` calls change tensor types and recompute layout.
4. `BeginWrite` creates the output file and writes:
   - Header (magic, version, counts)
   - Metadata bytes (copied from source)
   - Tensor directory (rebuilt)
   - Padding up to data offset
5. `WriteTensor` writes each tensor's data and its padding, in directory order.
6. `Finish` verifies all tensors were written and closes the file.
7. `Destroy` cleans up; removes half-written file if `BeginWrite` succeeded but `Finish` was not called.

## Identity Copy

`GgufCopy` provides a simple interface for byte-identical copying:

1. Open source with `TGgufFile.Open`
2. Create writer from source
3. `BeginWrite` to destination path
4. For each tensor: read bytes from source, write with `WriteTensor`
5. `Finish`
6. Free everything

With no retyping, output is byte-identical to source.
