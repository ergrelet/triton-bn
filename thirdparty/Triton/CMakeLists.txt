add_library(triton INTERFACE)
target_include_directories(triton INTERFACE SYSTEM "Triton/src/libtriton/includes/")
target_link_directories(triton INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(triton INTERFACE "triton_x64")

# Setup the config file and make it accessible as it should
file(COPY "config.hpp" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/includes/triton")
target_include_directories(triton INTERFACE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/includes/")
