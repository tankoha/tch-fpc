# T01 `qlora_inspect` external design

## Purpose

`qlora_inspect` inspects the structure of a GGUF file without loading its tensors. It prints a machine-readable summary of the header, key-value metadata, and tensor directory.

## Command line

```sh
qlora_inspect [--kv] [--tensors] FILE
```

- `FILE`: Path to the GGUF file (absolute or relative).
- `--kv`: Include key-value metadata in the output (one line per key).
- `--tensors`: Include tensor directory in the output (one line per tensor).

## Output (stdout, tab-separated)

1. **Summary lines** (always printed):
   ```
   version<TAB>{version}
alignment<TAB>{alignment}
kv<TAB>{kv_count}
tensors<TAB>{tensor_count}
data_offset<TAB>{data_offset}
data_end<TAB>{data_end}
file_size<TAB>{file_size}
   ```

2. **Per-type counts** (one line per tensor type present, ordered by type ID):
   ```
   type<TAB>{type_name}<TAB>{count}
   ```

3. **Key-value metadata** (only with `--kv`, in file order):
   ```
   kv<TAB>{key}<TAB>{type}
   ```
   - `{type}` is the value-type name (e.g., `uint32`, `float64`).
   - Arrays: `array<elem_type>[count]` (e.g., `array<string>[248320]`).

4. **Tensor directory** (only with `--tensors`, in file order):
   ```
   tensor<TAB>{name}<TAB>{type_name}<TAB>{dims}<TAB>{offset}<TAB>{bytes}
   ```
   - `{dims}`: Dimensions in GGUF order, separated by spaces (e.g., `1024 248320`).

## Exit codes

| Code | Condition |
|---|---|
| 0 | File is valid GGUF and `data_end = file_size`. |
| 1 | File is not valid GGUF (error message on stderr) OR `data_end ≠ file_size` (output lines printed, then error message). |
| 2 | Usage error (e.g., no argument, unrecognized option). |

## Errors (stderr)

- Invalid GGUF: `error: {message}` (e.g., `error: bad magic`).
- Truncated file (shorter than its last tensor): `error: ...` naming the tensor, nothing on stdout.
- Trailing bytes after the last tensor: all lines printed, then `error: data ends at {data_end}, file size {file_size}`.
- Usage: `Usage: qlora_inspect [--kv] [--tensors] FILE`.