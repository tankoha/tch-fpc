# T01 program flow — `TGgufFile.Open` read sequence

Based on `tasks/T01-facts.md` (authoritative) and
`docs/detailed-design/T01-detailed.md` (2.5–2.7). The flowchart is the
read sequence of `Open` — header, KV (metadata) reads, tensor directory
reads, alignment, padding, and the data section start; the file stays open
(`TFileStream`, kept for `ReadTensorBytes`), positioning is by absolute
file offset, and no data-section bytes are read by `Open`.

Error exits are marked by `✗` with the condition and the message (message
per `tasks/T01-facts.md`: `Error messages name what was wrong and where:
the key or tensor name and the byte position when known.`, unless the facts
fix an exact message or `RunInspectCLI` reports `error: ...`).

```mermaid
flowchart TD
    A([Open APath]) --> B["open TFileStream 'fmOpenRead or fmShareDenyNone'<br/>keep it open (ReadTensorBytes)"]
    B --> H1["read magic UInt32 at 0"]
    H1 --> H2{"magic = $46554747 ('G' 'G' 'U' 'F')"}
    H2 -- no --> EX1(["✗ raises EGgufError: bad magic (2.6/1)<br/>qlora_inspect: 'error: …' on stderr, exit 1"])
    H2 -- yes --> H3["read version UInt32 at 4"]
    H3 --> H4{version 2 or 3}
    H4 -- no --> EX2(["✗ raises EGgufError: version <> 2 or 3 (2.6/2)"])
    H4 -- yes --> H5["read n_tensors UInt64 at 8<br/>read n_kv UInt64 at 16 (KVStart = 24)"]
    H5 --> K0{"KV loop, i = 0..n_kv-1"}
    K0 -- more --> K1["read key: UInt64 count at Pos, then count bytes (string)"]
    K1 --> K2["read value type UInt32; store VType, ArrType, Count (1 for a scalar)"]
    K2 --> K3{value type id}
    K3 -- scalar 0..7, 10..12 --> K4a["read the scalar value -> IntVal / FloatVal / StrVal"]
    K3 -- 8 string --> K4b["read the string value -> StrVal"]
    K3 -- 9 array --> K5a["read element type UInt32, count UInt64 -> ArrType, Count"]
    K3 -- other --> EX5(["✗ raises EGgufError: unknown value type id (2.6/5)"])
    K5a --> K5b{element type}
    K5b -- array --> EX6(["✗ raises EGgufError: arrays of arrays (2.6/6)"])
    K5b -- scalar --> K5c["read count elements -> Strs / Ints / Floats"]
    K4a --> K6
    K4b --> K6{"key = 'general.alignment'?"}
    K5c --> K6
    K6 -- yes --> K7{uint32, non-zero, power of two}
    K7 -- no --> EX7(["✗ raises EGgufError: bad general.alignment (2.6/7)"])
    K7 -- yes --> K8["remember Alignment (else default 32)"]
    K6 -- no --> K8
    K8 --> K9["KV[i] stored; set KVEnd after the last value"]
    K9 --> K0
    K0 -- KV done --> T0{"tensor directory, i = 0..n_tensors-1<br/>run := 0"}
    T0 -- more --> T1["read name (string), n_dims UInt32"]
    T1 --> T2{n_dims 1..4}
    T2 -- no --> EX8(["✗ raises EGgufError: bad n_dims (2.6/8)"])
    T2 -- yes --> T3["read n_dims dims UInt64 (Dims[0] = row length)"]
    T3 --> T4["read tensor type UInt32, offset UInt64"]
    T4 --> T5{type in 0, 1, 8, 12, 13, 14, 30}
    T5 -- no --> EX9(["✗ raises EGgufError: unsupported tensor type N (2.4)"])
    T5 -- yes --> T6{Dims[0] multiple of block size}
    T6 -- no --> EX10(["✗ raises EGgufError: bad row length (2.4/2.6/8)"])
    T6 -- yes --> T7["NBytes := GgufTensorBytes (elements div block × block bytes)"]
    T7 --> T8{offset = run}
    T8 -- no --> EX11(["✗ raises EGgufError: wrong tensor offset (2.6/8)"])
    T8 -- yes --> T9["run := run + round-up NBytes to Alignment"]
    T9 --> T10{DataOffset + offset + NBytes <= FileSize}
    T10 -- no --> EX12(["✗ raises EGgufError: tensor past end of file (2.6/8)"])
    T10 -- yes --> T11{"end of last tensor = FileSize (well-formed)?<br/>else 'data ends at X, file size Y' in RunInspectCLI"}
    T11 --> T13["Tensor[i] stored (Name, NDims, Dims, TType, Offset, NBytes)"]
    T13 --> T0
    T0 -- tensors done --> D1["DataOffset := round-up (position after last tensor info) to Alignment"]
    D1 --> D2["padding to the alignment — skipped, not read"]
    D2 --> DOK(["Open done; stream kept open"]) --> D3["data section at DataOffset<br/>(ReadTensorBytes: seek DataOffset + Tensor[i].Offset, read NBytes)"]
    K5 -- read past end --> EX4(["✗ raises EGgufError: file ends early (2.6/10)"])
    T4 -- read past end --> EX4
    K1 -- read past end --> EX4
```

## Read sequence and positions

Everything is little-endian. `Pos` = the current absolute file offset;
each read advances it by the field's size. `T01-facts.md` fixes the
layout; the detail design (2.7) fixes the arithmetic.

| # | step | read at `Pos` (size) | sets / checks |
|---|---|---|---|
| 1 | open | — | `TFileStream` (`fmOpenRead or fmShareDenyNone`), kept open |
| 2 | magic | 0 (4) | `$46554747`, else ✗ 2.6/1 |
| 3 | version | 4 (4) | 2 or 3, else ✗ 2.6/2 |
| 4 | `n_tensors` / `n_kv` | 8 (8) / 16 (8) | ✗ on read past end (2.6/10) |
| 5 | KV × `n_kv` | from `KVStart = 24` | key string (count UInt64 + bytes, 2.6/9), value type (2.6/5), value; arrays: element type + count + elements (2.6/6); `general.alignment` checked and remembered (2.6/7, default 32); after the last value `KVEnd = Pos` |
| 6 | tensor directory × `n_tensors` | continuing `Pos` | name (2.6/9), `n_dims` 1..4 (2.6/8), `n_dims` × UInt64, type (2.6/8, table 2.4), offset (relative to the data section); per entry: `NBytes := GgufTensorBytes` (2.4), offset = running sum of earlier `round-up(bytes, Alignment)` (first = 0), `DataOffset + offset + NBytes <= FileSize` (2.6/8) |
| 7 | alignment + padding | not read | `Alignment` = `general.alignment` else 32 (2.6/7); `DataOffset := round-up(Pos after the last tensor info, Alignment)` (2.7); the padding bytes are skipped without being read |
| 8 | data section | `DataOffset` onward | read on demand by `ReadTensorBytes`: seek `DataOffset + Tensor[i].Offset`, read `Tensor[i].NBytes` exactly (2.10) |

## Error exits (condition → message)

| # | node | condition (facts order) | message |
|---|---|---|---|
| 1 | EX1 | magic ≠ 4 bytes `'G' 'G' 'U' 'F'` (`$46554747`) | `EGgufError`; CLI: `error: ...` on stderr, exit 1, nothing on stdout |
| 2 | EX2 | `version UInt32` not 2 or 3 — `anything else is an error` | ditto |
| 3 | EX4 | `A file that ends early (any read past the end) is an error.` (KV or tensor-directory reads) | ditto |
| 4 | EX5 | value type not `0 uint8 … 12 float64` — `Any other number is an error.` | ditto; message names the key and the byte position when known (2.6/11) |
| 5 | EX6 | `Arrays of arrays are an error (upstream does not allow them).` | ditto (element type `array`, 2.6/6) |
| 6 | EX7 | `general.alignment` present but not `uint32, non-zero and a power of two, else error` (absent → 32) | ditto (key `general.alignment`) |
| 7 | EX8 | `n_dims` outside `1..4` | ditto (tensor name + position) |
| 8 | EX9 | tensor id outside `0, 1, 8, 12, 13, 14, 30` — `Any other id is an error "unsupported tensor type N"` | `unsupported tensor type N` |
| 9 | EX10 | `The first dimension (ne[0], the row length) must be a multiple of the block size` — violated | ditto (names the tensor and the offending `ne[0]` / block size) |
| 10 | EX11 | `tensor i's offset must equal the sum over the tensors before it of their bytes, each rounded up to the alignment` — violated (the first offset is 0) | ditto (names the tensor) |
| 11 | EX12 | `The end of a tensor (data start + offset + bytes) must not exceed the file size.` — violated | ditto (names the tensor) |
| 12 | T11 | Open passed but `data_end <> file_size` | not in `Open`: `RunInspectCLI` prints all lines, then `error: data ends at X, file size Y` on stderr, exit 1 |

## Notes

- Branches 7–12 in the flowchart mirror the check order of
  `docs/detailed-design/T01-detailed.md` 2.6 one-to-one; each branch is a
  UT-case candidate (T01a step 4).
- Implementation (T01b2): the tensor directory is read first (every
  entry's name, `n_dims` — checked 1..4 as it is read — dimensions, type,
  offset), then `DataOffset` is computed (2.7), and only then are the
  per-entry checks run in entry order (`GgufTensorBytes` type + row length,
  offset = `run`, extent). This is forced by 2.7: `DataOffset` is only
  known after the last tensor info, so the extent check cannot happen
  while its entry is being read. The read sequence — which fields, in
  which order, at which byte positions — is exactly as the flowchart
  shows; `Open` reads the data section not at all.
- `ReadTensorBytes` is not in the chart (it is not part of the read
  sequence of `Open`); its only check is the one inherited from Open —
  `DataOffset + offset + NBytes <= FileSize` — plus `A file that ends
  early (any read past the end) is an error.` during the read (2.10).
- `FindKey` / `FindTensor` are lookups in the stored arrays
  (`exact, case-sensitive`, `-1 if absent`) — no file reads, checked and
  errors: none.
- `qlora_inspect` maps the exits: usage error → exit 2; `EGgufError` →
  exit 1 with `error: ...` and nothing on stdout; `data_end <>
  file_size` → exit 1 with all lines already printed; otherwise exit 0
  (2.12).

## Change history

| Date | Change | Reason |
|---|---|---|
| 2026-11-18 | Initial program flow | T01a step 3; from `tasks/T01-facts.md` and the T01 detailed design |
| 2026-11-20 | Notes: document the two-phase tensor-directory processing of the T01b2 implementation (read pass, `DataOffset` per 2.7, then the per-entry checks) | the stage-2 code validates the extent only after `DataOffset` is known; the read sequence and the checks themselves are unchanged |

