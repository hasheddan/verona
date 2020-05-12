//===- VeronaOps.h - Verona dialect ops -----------------*- C++ -*-===//
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.
// See https://github.com/microsoft/verona/blob/master/LICENSE for license
// information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef VERONA_VERONAOPS_H
#define VERONA_VERONAOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace verona {

#define GET_OP_CLASSES
#include "dialect/VeronaOps.h.inc"

} // namespace verona
} // namespace mlir

#endif // VERONA_VERONAOPS_H
