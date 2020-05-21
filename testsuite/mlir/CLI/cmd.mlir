// RUN: verona-mlir %s -o - | verona-mlir | FileCheck %s
// RUN: verona-mlir --emit-mlir %s -o - | verona-mlir | FileCheck %s
// RUN: verona-mlir --emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

module {
    // CHECK-LABEL: func @bar() -> i32
    // LLVM-LABEL: define i32 @bar()
    func @bar() -> i32 {
        // CHECK: constant 1
        %0 = constant 1 : i32
        // CHECK: return
        // LLVM: ret i32 1
        return %0 : i32
    }
}
