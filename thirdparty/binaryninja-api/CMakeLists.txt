# set(HEADLESS TRUE)
# add_subdirectory("binaryninja-api" EXCLUDE_FROM_ALL)

include(FetchContent)

if(TRITON_BN_BINARYNINJA_CHANNEL STREQUAL "stable")
    set(_binexport_binaryninjacore_suffix "_stable")
    set(_binexport_binaryninja_git_tag
        "14905bd51979f4f55dfe3e0b299d9a33d9343ef6") # 2022-05-26
else()
    set(_binexport_binaryninjacore_suffix "")
    set(_binexport_binaryninja_git_tag
        "4bb510c2e4456606e03acfc6970bf8697156769e") # 2022-06-02
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