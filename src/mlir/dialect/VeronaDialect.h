//===- VeronaDialect.h - Verona dialect -----------------*- C++ -*-===//
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.
// See https://github.com/microsoft/verona/blob/master/LICENSE for license
// information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef VERONA_VERONADIALECT_H
#define VERONA_VERONADIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace verona {

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

} // namespace verona
} // namespace mlir

#endif // VERONA_VERONADIALECT_H
