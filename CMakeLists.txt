cmake_minimum_required(VERSION 3.18)

project(triton-bn)

# Dependencies
add_subdirectory("thirdparty")

# Plugin module
add_library(triton_bn_plugin SHARED "src/main.cc")
target_link_libraries(triton_bn_plugin PRIVATE binaryninjaapi binaryninjaui triton)
set_target_properties(triton_bn_plugin PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Tests
add_subdirectory("tests")
