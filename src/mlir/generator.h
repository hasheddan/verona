// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR.h"
#include "symbol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <peglib.h>
#include <string>

namespace mlir::verona
{
  struct Generator
  {
    Generator() : builder(&context)
    {
      context.allowUnregisteredDialects();
    }

    // Read AST/MLIR into the opaque MLIR format
    void readAST(const ::ast::Ast& ast);
    void readMLIR(const std::string& filename);

    // Transform the opaque MLIR format into Verona dialect and LLVM IR.
    mlir::ModuleOp
    emitMLIR(const llvm::StringRef filename = "", unsigned optLevel = 0);
    std::unique_ptr<llvm::Module>
    emitLLVM(const llvm::StringRef filename = "", unsigned optLevel = 0);

    using Types = llvm::SmallVector<mlir::Type, 4>;
    using Values = llvm::SmallVector<mlir::Value, 4>;

  private:
    // MLIR module, builder and context.
    mlir::OwningModuleRef module;
    mlir::OpBuilder builder;
    mlir::MLIRContext context;

    // Symbol tables for variables, functions and classes.
    SymbolTableT symbolTable;
    FunctionTableT functionTable;
    TypeTableT typeTable;

    // Get location of an ast node
    mlir::Location getLocation(const ::ast::Ast& ast);

    // Parses a module, the global context.
    void parseModule(const ::ast::Ast& ast);

    // Parses a function, from a top-level (module) view.
    mlir::FuncOp parseProto(const ::ast::Ast& ast);
    mlir::FuncOp parseFunction(const ::ast::Ast& ast);

    // Parses a global variable, from a top-level (module) view.
    mlir::Value parseGlobal(const ::ast::Ast& ast);

    // Recursive type parser, gathers all available information on the type
    // and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

    // Declares/Updates a variable.
    void declareVariable(llvm::StringRef name, mlir::Value val);
    void updateVariable(llvm::StringRef name, mlir::Value val);

    // Generic block/node parser, calls other parse functions to handle each
    // individual type. Block returns last value, for return.
    mlir::Value parseBlock(const ::ast::Ast& ast);
    mlir::Value parseNode(const ::ast::Ast& ast);
    mlir::Value parseValue(const ::ast::Ast& ast);

    // Specific parsers (there will be more).
    mlir::Value parseAssign(const ::ast::Ast& ast);
    mlir::Value parseCall(const ::ast::Ast& ast);
    mlir::Value parseLet(const ::ast::Ast& ast);

    // Wrappers for unary/binary operands
    mlir::Value genOperation(
      mlir::Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Value> ops,
      mlir::Type retTy);
  };
}
