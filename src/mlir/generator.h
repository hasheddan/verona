// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
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

    // The symbol table has all declared symbols (with the original node
    // and the MLIR ounterpart) in a scope. Creating a new scope makes
    // all future insertions happen at that level, destroying it pops
    // the scope out of the stack.
    struct Symbol
    {
      mlir::Value val;
      ::ast::Ast* node;
    };
    using SymbolTableT = llvm::ScopedHashTable<llvm::StringRef, Symbol>;
    // This is the root scope. New scopes are created by creating a new
    // local variable like:
    //   SymbolTableT var_scope(symbolTable);
    // The destructor pops the scope automatically.
    SymbolTableT symbolTable;

    // Get location of an ast node
    mlir::Location getLocation(const ::ast::Ast& ast);

    // Parses a module, the global context.
    void parseModule(const ::ast::Ast& ast);

    // Parses a function, from a top-level (module) view.
    mlir::FuncOp parseFunction(const ::ast::Ast& ast);

    // Parses a global variable, from a top-level (module) view.
    mlir::Value parseGlobal(const ::ast::Ast& ast);

    // Recursive type parser, gathers all available information on the type
    // and sub-types, modifiers, annotations, etc.
    mlir::Type parseType(const ::ast::Ast& ast);

    // Declare a program variable on the current scope (via symbolTable).
    void declareVariable(const ::ast::Ast& ast, mlir::Value);

    // Declares a compiler variable, for auto-gen code.
    void declareVariable(llvm::StringRef name, mlir::Value);

    // Generic node parser, calls other parse functions to handle each
    // individual type.
    mlir::Value parseNode(const ::ast::Ast& ast);

    // Specific parsers (there will be more).
    mlir::Value parseOperation(const ::ast::Ast& ast);
    mlir::Value parseCall(const ::ast::Ast& ast);
    mlir::Value parseLet(const ::ast::Ast& ast);
    mlir::Value parseReturn(const ::ast::Ast& ast);
  };
}
