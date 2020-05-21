// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir::verona {

namespace VeronaTypes {
// Needs to be an enum (not an enum class) because 'kindof' methods compare
// unsigned values and not class values.
enum Kind {
  Opaque
};
} // namespace VeronaTypes

#include "dialect/VeronaOpsDialect.h.inc"

class OpaqueType : public Type::TypeBase<OpaqueType, Type> {
  std::string desc;
public:
  using Base::Base;

  static OpaqueType get(MLIRContext *context, StringRef desc) {
    auto type = Base::get(context, VeronaTypes::Kind::Opaque);
    type.desc = desc;
    return type;
  }

  StringRef getDescription() const {
    return desc;
  }

  static bool kindof(unsigned kind) {
    return kind == VeronaTypes::Opaque;
  }
};

} // namespace mlir::verona
