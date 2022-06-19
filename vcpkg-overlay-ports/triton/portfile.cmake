vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO JonathanSalwan/Triton
  REF 9e6eb2492c7b66e8e1ccc14d75eb89705d108746
  SHA512 1e980c27c0240854c91f374e29c873016641833754b0d084d5caae3f73b572024d14a03f661e11bd512cf44dd5f92d9bd826af1c407a414502c8dca2f3256174
  PATCHES
    001-fix-dependency-z3.patch
    003-fix-python.patch
)

string(COMPARE NOTEQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" DYNAMICLIB)
string(COMPARE EQUAL "${VCPKG_CRT_LINKAGE}" "static" STATICCRT)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        "python" PYTHON_BINDINGS
)

set(ADDITIONAL_OPTIONS )
if(PYTHON_BINDINGS)
    vcpkg_find_acquire_program(PYTHON3)
    list(APPEND ADDITIONAL_OPTIONS
        -DPYTHON_EXECUTABLE=${PYTHON3}
        )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DZ3_INTERFACE=ON
        -DTRITON_BOOST_INTERFACE=OFF
        -DBUILD_SHARED_LIBS=${DYNAMICLIB}
        -DMSVC_STATIC=${STATICCRT}
        -DBUILD_EXAMPLES=OFF
        -DENABLE_TEST=OFF
        ${ADDITIONAL_OPTIONS}
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()

# Move cmake configs
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${PORT})

# Remove duplicate files
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE.txt" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
