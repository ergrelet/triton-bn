cmake_minimum_required(VERSION 3.18)

set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg-overlay-ports")

project(triton-bn)

option(TRITON_BN_BUILD_TESTS "Build test executables" OFF)

set(TRITON_BN_BINARYNINJA_CHANNEL "stable" CACHE
    STRING "Binary Ninja channel, either 'stable' or 'dev'")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX_EXTENSIONS OFF)

find_package(triton CONFIG REQUIRED)

# Binary Ninja
add_subdirectory("thirdparty")

# Plugin module
add_library(triton_bn_plugin SHARED
    "src/main.cc"
    "src/meta_basic_block.h"
    "src/meta_basic_block.cc"
    "src/commands.h"
    "src/commands.cc"
)
target_link_libraries(triton_bn_plugin PRIVATE BinaryNinja::API triton::triton)

# Tests
if(TRITON_BN_BUILD_TESTS)
    add_subdirectory("tests")
endif()
