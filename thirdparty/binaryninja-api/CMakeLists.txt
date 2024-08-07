include(FetchContent)

if(TRITON_BN_BINARYNINJA_CHANNEL STREQUAL "stable")
    set(_triton_bn_binaryninjacore_suffix "_stable")
    set(_triton_bn_binaryninja_git_tag "v4.0.4958-stable")
else()
    set(_triton_bn_binaryninjacore_suffix "")
    set(_triton_bn_binaryninja_git_tag
        "68ac3ce7b8442c4a5bcc8259dd1fa156cdcefa03") # 2024-04-05
endif()
FetchContent_Declare(binaryninjaapi
    GIT_REPOSITORY https://github.com/Vector35/binaryninja-api.git
    GIT_TAG        ${_triton_bn_binaryninja_git_tag}
)
FetchContent_GetProperties(binaryninjaapi)
if(NOT binaryninjaapi_POPULATED)
    FetchContent_Populate(binaryninjaapi)  # For binaryninjaapi_SOURCE_DIR
endif()
add_library(binaryninjacore SHARED
    binaryninjacore${_triton_bn_binaryninjacore_suffix}.cc
)
set_target_properties(binaryninjacore PROPERTIES
    SOVERSION 1
)
target_include_directories(binaryninjacore PRIVATE
    "${binaryninjaapi_SOURCE_DIR}"
)
set(CORE_LIBRARY binaryninjacore)
set(BN_CORE_LIBRARY "${CORE_LIBRARY}")
set(HEADLESS TRUE)
if(binaryninjaapi_POPULATED)
    add_subdirectory("${binaryninjaapi_SOURCE_DIR}" "${binaryninjaapi_BINARY_DIR}")
endif()
add_library(BinaryNinja::API ALIAS binaryninjaapi)