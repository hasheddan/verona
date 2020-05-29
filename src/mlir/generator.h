// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR.h"

#include <string>

namespace mlir::verona
{
  class Generator
  {
    mlir::OwningModuleRef module;
    mlir::OpBuilder builder;
    mlir::MLIRContext context;
    mlir::Location UNK;

  public:
    Generator() : builder(&context), UNK(builder.getUnknownLoc()) {}
    void readAST(::ast::Ast& ast);
    void readMLIR(std::string& filename);
    mlir::ModuleOp
    emitMLIR(llvm::StringRef filename = "", unsigned optLevel = 0);
    std::unique_ptr<llvm::Module>
    emitLLVM(llvm::StringRef filename = "", unsigned optLevel = 0);
  };
}
