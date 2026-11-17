# Devices

## Writing device-agnostic code
- Choose once: `Dev := TDevice.CudaIfAvailable;` (`TDevice.Cpu`,
  `TDevice.Cuda(I)`). Create the store on it (`TVarStore.Create(Dev)`) and
  move inputs with `X.&to(Dev)`.
- Every tensor of one op must be on the same device; otherwise libtorch
  raises `ETorchNativeError`.
- Read results element-wise on the CPU: `T.&to(TDevice.Cpu).DoubleValue([..])`.
- `T.Device.ToString` is `cpu`, `cuda:0`, ...
- `CudaIsAvailable`, `CudaDeviceCount`, `CudaManualSeed`, `CudaSynchronize`.

## ROCm (AMD)
- On a ROCm (HIP) build of libtorch "Cuda" means the AMD GPU: same code,
  same names.
- An integrated GPU counts as a device too; pick the discrete one(s) with
  `HIP_VISIBLE_DEVICES=0` (or `0,1`) in the environment of the program.
- `GpuBackend` returns `gbRocm`; `CudnnIsAvailable` is False (MIOpen does
  not stand in) and `GpuRuntimeVersion` and `CudnnVersion` raise.
- The ROCm libtorch zip has no symlinks: after unpacking, link seven
  libraries to their SONAMEs and `libnuma.so` to `libnuma.so.1` (DESIGN 4.8
  gives the one-line command). Without them the program exits at start-up.
- At start-up the ROCm build prints `(null): No such file or directory`
  four times on stderr (optional RDMA libraries); harmless.
- Do not train on the GPU that drives the display: a long kernel can hang
  the desktop's graphics ring.

## Builds
One libtorch directory and one `build/libtch-<variant>` per variant (CPU,
ROCm, CUDA); a program is linked against one of them by its rpath
(`reference/build.md`). `tests/gpu/run.sh` checks a GPU build.

Source: DESIGN 4.8.
