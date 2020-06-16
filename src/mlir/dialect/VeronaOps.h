// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::verona
{
#define GET_OP_CLASSES
#include "dialect/VeronaOps.h.inc"

} // namespace mlir::verona
