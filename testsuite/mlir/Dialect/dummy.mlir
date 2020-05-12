// RUN: verona-opt %s | verona-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = verona.foo %{{.*}} : i32
        %res = verona.foo %0 : i32
        %foo = "verona.test"(%res) : (i32) -> (!verona<"U64_imm">)
        return
    }
}
