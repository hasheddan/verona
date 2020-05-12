//===- VeronaDialect.td - Verona dialect -----------*- tablegen -*-===//
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.
// See https://github.com/microsoft/verona/blob/master/LICENSE for license
// information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef VERONA_DIALECT
#define VERONA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Verona dialect definition.
//===----------------------------------------------------------------------===//

def Verona_Dialect : Dialect {
    let name = "verona";
    let summary = "A prototype Verona language MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "verona";
}

//===----------------------------------------------------------------------===//
// Base verona operation/type definitions.
//===----------------------------------------------------------------------===//

class Verona_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Verona_Dialect, mnemonic, traits>;

def Verona_OpaqueType : DialectType<Verona_Dialect,
    CPred<"$_self.isa<::mlir::verona::OpaqueType>()">, "opaque type">,
    BuildableType<"$_builder.getType<::mlir::verona::OpaqueType>()"> {
  let typeDescription = [{
    `verona.opaque_type` handles strings defining static or dynamic types that
    haven't been converted into full Verona types yet. It is valid to have
    opaque types at the beginning, but before lowering to Std/Loop dialects
    all types must have been converted.
  }];
}

#endif // VERONA_DIALECT
