# Modified for tch-fpc from tch-rs 4227b89b72059b0651ff83a38637693574e0a2d6 (torch-sys/libtch):
# shared library instead of static, C++17 (required by libtorch 2.x), and a
# manual-link mode for GPU archives (DESIGN.md 4.8): -DTCH_LIBTORCH_DIR=<dir>
# links <dir>/lib directly instead of find_package(Torch), whose ROCm
# variant needs the ROCm SDK's cmake files.
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

set(TCH_LIBTORCH_DIR "" CACHE PATH
  "libtorch root; when set, link it directly instead of find_package(Torch)")

add_library(tch SHARED torch_api.cpp torch_api_generated.cpp)
if(TCH_LIBTORCH_DIR)
  target_include_directories(tch PRIVATE ${TCH_LIBTORCH_DIR}/include
    ${TCH_LIBTORCH_DIR}/include/torch/csrc/api/include)
  target_compile_definitions(tch PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
  # the umbrella libtorch.so pulls in the GPU backend (libtorch_hip.so /
  # libtorch_cuda.so); --as-needed would drop it
  target_link_libraries(tch -Wl,--no-as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch.so
    -Wl,--as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch_cpu.so
    ${TCH_LIBTORCH_DIR}/lib/libc10.so)
  set_target_properties(tch PROPERTIES BUILD_RPATH ${TCH_LIBTORCH_DIR}/lib)
else()
  find_package(Torch REQUIRED)
  target_link_libraries(tch "${TORCH_LIBRARIES}")
endif()
set_property(TARGET tch PROPERTY CXX_STANDARD 17)
install(TARGETS tch DESTINATION .)
