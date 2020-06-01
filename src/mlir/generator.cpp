// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#include "generator.h"

#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"
#include "dialect/VeronaDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
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
  // Helpers
  const ::ast::Ast& findNode(const ::ast::Ast& node, unsigned int tag) {
    assert(!node->is_token && "Bad node");
    // FIXME: Is there a better way of doing this?
    for (auto sub: node->nodes)
    {
      if (sub->tag == tag)
        return sub;
    }
    llvm_unreachable("Sub-node not found");
  }

  // Generation
  void Generator::parseModule(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Module && "Bad node");
    module = mlir::ModuleOp::create(UNK);
    // TODO: Support more than just functions at the module level
    for (auto fun: ast->nodes)
    {
      module->push_back(parseFunction(fun));
    }
  }

  mlir::FuncOp Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Function && "Bad node");
    // Get function name
    auto name = findNode(findNode(ast, NodeType::FuncName), NodeType::ID);

    // Function type from signature
    llvm::SmallVector<mlir::Type, 1> argTypes;
    mlir::Type voidTy{};
    auto funcTy = builder.getFunctionType(argTypes, voidTy);

    // Create function
    auto func = mlir::FuncOp::create(UNK, name->name, funcTy);

    // TODO: lower body
    //auto &entryBlock = *func.addEntryBlock();
    //builder.setInsertionPointToStart(&entryBlock);
    //auto last = buildNode(def->getImpl());
    //builder.create<mlir::ReturnOp>(UNK, last);

    return func;
  }
}
