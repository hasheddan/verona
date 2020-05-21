# Verona MLIR Dialect

This is a prototype for the Verona MLIR dialect.

## Building

Build Verona with additional CMake options:

* __-DEXPERIMENTAL_MLIR_COMPILER=True__
* __-DLLVM_BUILD=/absolute/path/to/llvm-projects/build__

Build LLVM with the following options:

```bash
$ $ cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON \
        -DLLVM_REQUIRES_EH=ON \
        -DLLVM_REQUIRES_RTTI=ON
$ ninja check-mlir
```

_Note: EH/RTTI is needed by Verona, so you **must** compile LLVM with it, too_
