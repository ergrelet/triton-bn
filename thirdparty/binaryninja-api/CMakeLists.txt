include(FetchContent)

if(TRITON_BN_BINARYNINJA_CHANNEL STREQUAL "stable")
    set(_binexport_binaryninjacore_suffix "_stable")
    set(_binexport_binaryninja_git_tag
        "919384bb2bb9216e000750a00793549ef7a46d87") # 2023-01-17
else()
    set(_binexport_binaryninjacore_suffix "")
    set(_binexport_binaryninja_git_tag
        "9d8bc846bd213407fb1a7a19af2a96f17501ac3b") # 2023-02-15
endif()
FetchContent_Declare(binaryninjaapi
    GIT_REPOSITORY https://github.com/Vector35/binaryninja-api.git
    GIT_TAG        ${_binexport_binaryninja_git_tag}
    GIT_SUBMODULES "docs" # Workaround for CMake #20579
)
FetchContent_GetProperties(binaryninjaapi)
if(NOT binaryninjaapi_POPULATED)
    FetchContent_Populate(binaryninjaapi)  # For binaryninjaapi_SOURCE_DIR
endif()
add_library(binaryninjacore SHARED
    binaryninjacore${_binexport_binaryninjacore_suffix}.cc
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