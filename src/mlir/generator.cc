// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#include "ast-utils.h"
#include "generator.h"

#include "dialect/VeronaDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::verona
{
  // ===================================================== Public Interface
  void Generator::readAST(const ::ast::Ast& ast)
  {
    parseModule(ast);
    // On error, dump module for debug purposes
    if (mlir::failed(mlir::verify(*module)))
    {
      module->dump();
      throw std::runtime_error("Failed to parse Verona file");
    }
  }

  void Generator::readMLIR(const std::string& filename)
  {
    // Read an MLIR file
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> srcOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);

    // Setup source manager and parse
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*srcOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, builder.getContext());
  }

  mlir::ModuleOp
  Generator::emitMLIR(llvm::StringRef filename, unsigned optLevel)
  {
    // On error, dump module for debug purposes
    if (mlir::failed(mlir::verify(*module)))
    {
      module->dump();
      throw std::runtime_error("Failed to parse MLIR file");
    }
    if (filename.empty())
      return module.get();

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      throw std::runtime_error("Failed open output file");

    module->print(out);
    return module.get();
  }

  std::unique_ptr<llvm::Module>
  Generator::emitLLVM(llvm::StringRef filename, unsigned optLevel)
  {
    // The lowering "pass manager"
    mlir::PassManager pm(&context);
    if (optLevel > 0)
    {
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      mlir::OpPassManager& optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
    }
    pm.addPass(mlir::createLowerToLLVMPass());

    // First lower to LLVM dialect
    if (mlir::failed(pm.run(module.get())))
    {
      module->dump();
      throw std::runtime_error("Failed to lower to LLVM dialect");
    }

    // Then lower to LLVM IR
    auto llvm = mlir::translateModuleToLLVMIR(module.get());
    if (!llvm)
      throw std::runtime_error("Failed to lower to LLVM IR");

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      throw std::runtime_error("Failed open output file");

    llvm->print(out, nullptr);
    return llvm;
  }

  // ===================================================== AST -> MLIR
  mlir::Location Generator::getLocation(const ::ast::Ast& ast)
  {
    return builder.getFileLineColLoc(Identifier::get(ast->path, &context), ast->line, ast->column);
  }

  mlir::Type Generator::parseType(const ::ast::Ast& ast)
  {
    auto dialect = Identifier::get("type", &context);
    auto desc = getTypeDesc(ast);
    return mlir::OpaqueType::get(dialect, desc, &context);
  }

  void Generator::parseModule(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Module && "Bad node");
    module = mlir::ModuleOp::create(getLocation(ast));
    // TODO: Support more than just functions at the module level
    for (auto fun: ast->nodes)
      module->push_back(parseFunction(fun));
  }

  mlir::FuncOp Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Function && "Bad node");

    // Function type from signature
    Types types;
    auto args = getFunctionArgs(ast);
    for (auto argTy: args)
      types.push_back(parseType(getType(argTy).lock()));
    auto retTy = parseType(getFunctionType(ast).lock());
    auto funcTy = builder.getFunctionType(types, retTy);

    // Create function
    auto name = getFunctionName(ast);
    auto func = mlir::FuncOp::create(getLocation(ast), name, funcTy);

    // TODO: lower body
    auto body = getFunctionBody(ast);
    //auto &entryBlock = *func.addEntryBlock();
    //builder.setInsertionPointToStart(&entryBlock);
    //auto last = buildNode(def->getImpl());
    //builder.create<mlir::ReturnOp>(getLocation(ast), last);

    return func;
  }
}
