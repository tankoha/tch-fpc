# Modified for tch-fpc from tch-rs 4227b89b72059b0651ff83a38637693574e0a2d6 (torch-sys/libtch):
# shared library instead of static, C++17 (required by libtorch 2.x).
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(tch)

find_package(Torch REQUIRED)

add_library(tch SHARED torch_api.cpp torch_api_generated.cpp)
target_link_libraries(tch "${TORCH_LIBRARIES}")
set_property(TARGET tch PROPERTY CXX_STANDARD 17)
install(TARGETS tch DESTINATION .)
