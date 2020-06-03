// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::verona
{
  // The symbol table has all declared symbols (with the original node
  // and the MLIR ounterpart) in a scope. Creating a new scope makes
  // all future insertions happen at that level, destroying it pops
  // the scope out of the stack.

  // New scopes are created by creating a new local variable like:
  //   SymbolScopeT scope(symbolTable);
  // The destructor pops the scope automatically.

  // We cannot use LLVM's ADT/ScopedHashTable like MLIR's Toy example because
  // that coped hash table does not allow redefinition, which is a problem when
  // declaring variables with a type only and then assigning values later.
  template<class T>
  class ScopedTable
  {
    using MapTy = std::map<std::string, T>;
    std::vector<MapTy> stack;
    int currentScope;

  public:
    ScopedTable() : currentScope(-1)
    {
      pushScope();
    }
    ~ScopedTable()
    {
      popScope();
      assert(stack.empty() && currentScope == -1);
    }
    bool insert(llvm::StringRef key, T value)
    {
      auto& frame = stack[currentScope];
      if (frame.count(key.str()))
        return false;
      auto res = frame.emplace(key, value);
      return res.second;
    }
    T lookup(llvm::StringRef key)
    {
      for (int i = currentScope; i >= 0; i--)
      {
        auto& frame = stack[i];
        if (frame.count(key.str()))
          return frame[key.str()];
      }
      // Use inScope for this not to happen
      llvm_unreachable("Symbol not found");
    }
    bool inScope(llvm::StringRef key)
    {
      auto frame = stack[currentScope];
      return frame.count(key.str());
    }
    bool update(llvm::StringRef key, T value)
    {
      auto& frame = stack[currentScope];
      if (!frame.count(key.str()))
        return insert(key, value);
      // FIXME: Check types are compatible
      frame[key.str()] = value;
      return true;
    }
    void pushScope()
    {
      currentScope++;
      stack.emplace_back();
    }
    void popScope()
    {
      currentScope--;
      stack.resize(stack.size() - 1);
    }
  };

  // FIXME: This is a hack to control scope. We can do better.
  template<class T>
  class ScopedTableScope
  {
    ScopedTable<T>& table;

  public:
    ScopedTableScope(ScopedTable<T>& table) : table(table)
    {
      table.pushScope();
    }
    ~ScopedTableScope()
    {
      table.popScope();
    }
  };

  // Variable symbols. New scopes should be created when entering classes,
  // functions, lexical blocks, lambdas, etc.
  using SymbolTableT = ScopedTable<mlir::Value>;
  using SymbolScopeT = ScopedTableScope<mlir::Value>;

  // Function Symbols. New scopes should be created when entering classes
  // and sub-classes. Modules too, if we allow more than one per file.
  using FunctionTableT = ScopedTable<mlir::FuncOp>;
  using FunctionScopeT = ScopedTableScope<mlir::FuncOp>;
}
