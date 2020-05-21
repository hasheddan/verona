// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "VeronaDialect.h"
#include "VeronaOps.h"

using namespace mlir;
using namespace mlir::verona;

//===----------------------------------------------------------------------===//
// Verona dialect.
//===----------------------------------------------------------------------===//

VeronaDialect::VeronaDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "dialect/VeronaOps.cpp.inc"
      >();
  addTypes<OpaqueType>();
  allowUnknownOperations();
  allowUnknownTypes();
}

Type VeronaDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) {
    return Type();
  } else {
    return OpaqueType::get(getContext(), keyword);
  }

  parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
  return Type();
}

void VeronaDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch(type.getKind()) {
    case VeronaTypes::Opaque: {
      auto oTy = type.cast<OpaqueType>();
      os << "!verona<\"" << oTy.getDescription() << "\">";
      return;
    }
    default:
      llvm_unreachable("unexpected 'verona' type kind");
  }
}
