//===- VeronaOps.td - Verona dialect ops -----------*- tablegen -*-===//
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.
// See https://github.com/microsoft/verona/blob/master/LICENSE for license
// information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef VERONA_OPS
#define VERONA_OPS

include "VeronaDialect.td"
include "mlir/Interfaces/SideEffects.td"

def Verona_FooOp : Verona_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `verona.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = verona.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // VERONA_OPS
