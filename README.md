# triton-bn ![Static Badge](https://img.shields.io/badge/Binary_Ninja_API-v4.0.x-blue)

`triton-bn` is a small Binary Ninja plugin that can be used to apply
[Triton](https://github.com/jonathansalwan/Triton)'s dead store eliminitation
pass on basic blocks or functions.

This plugin may also serve as a base for people that would want to play with
Triton inside of Binary Ninja.

## How to Build

On Windows:
```
$ git clone --recurse-submodule https://github.com/ergrelet/triton-bn.git && cd triton-bn
$ ./vcpkg/bootstrap-vcpkg.bat
$ cmake -B build -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
$ cmake --build build --config Release -- -maxcpucount
```

On Linux distributions:
```
$ git clone --recurse-submodule https://github.com/ergrelet/triton-bn.git && cd triton-bn
$ ./vcpkg/bootstrap-vcpkg.sh
$ cmake -B build -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake
$ cmake --build build -- -j$(nproc)
```

## How to Install

Check out the official Binary Ninja documentation to know where to copy the
files:
[Using Plugins](https://docs.binary.ninja/guide/plugins.html)


## Know Limitations
* Doesn't support ARM64 binaries
* Instructions  that use RIP-relative addressing aren't relocated properly after simplification
