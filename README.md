# triton-bn

## How to Build

On Windows:
```
$ ./vcpkg/bootstrap-vcpkg.bat
$ cmake -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
$ cmake --build build --config Release -- -maxcpucount
```

On Linux distributions:
```
$ ./vcpkg/bootstrap-vcpkg.sh
$ cmake -B build -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
$ cmake --build build -- -j$(nproc)
```
