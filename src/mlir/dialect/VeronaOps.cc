// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "VeronaOps.h"

#include "VeronaDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::verona
{
#define GET_OP_CLASSES
#include "dialect/VeronaOps.cpp.inc"

} // namespace mlir::verona
