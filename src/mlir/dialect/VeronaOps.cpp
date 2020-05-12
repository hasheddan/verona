//===- VeronaOps.cpp - Verona dialect ops ---------------*- C++ -*-===//
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.
// See https://github.com/microsoft/verona/blob/master/LICENSE for license
// information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "VeronaOps.h"
#include "VeronaDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace verona {
#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"
} // namespace verona
} // namespace mlir
