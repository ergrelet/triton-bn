if(WIN32)
    add_library(triton INTERFACE)
    target_include_directories(triton INTERFACE SYSTEM "Triton/src/libtriton/includes/")
    target_link_directories(triton INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(triton INTERFACE "triton_x64")

    # Setup the config file and make it accessible as it should
    file(COPY "config.hpp" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/includes/triton")
    target_include_directories(triton INTERFACE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/includes/")
else()
    set(BOOST_INTERFACE OFF CACHE BOOL "Use Boost as multiprecision library" FORCE)
    # Note: Workaround needed on Fedora
    set(Z3_PKGCONF_INCLUDE_DIRS "/usr/include/z3")
    add_subdirectory("Triton" EXCLUDE_FROM_ALL)
    
    # Needed to link triton as library, because triton do not propagate its
    # public include directories
    target_include_directories(triton PUBLIC "${Z3_PKGCONF_INCLUDE_DIRS}")
endif()