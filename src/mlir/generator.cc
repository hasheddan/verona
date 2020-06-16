// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator.h"

#include "ast-utils.h"
#include "dialect/VeronaDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
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
      throw std::runtime_error("MLIR verification failed from Verona file");
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
      throw std::runtime_error("MLIR verification failed from MLIR file");
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
    return builder.getFileLineColLoc(
      Identifier::get(ast->path, &context), ast->line, ast->column);
  }

  mlir::Type Generator::parseType(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::OfType && "Bad node");
    auto desc = getTypeDesc(ast);
    if (desc.empty())
      return builder.getNoneType();

    // If type is in the alias table, get it
    if (typeTable.inScope(desc))
      return typeTable.lookup(desc);

    // Else, insert into the table and return
    auto dialect = Identifier::get("type", &context);
    auto type = mlir::OpaqueType::get(dialect, desc, &context);
    typeTable.insert(desc, type);
    return type;
  }

  void Generator::parseModule(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Module && "Bad node");
    module = mlir::ModuleOp::create(getLocation(ast));
    // TODO: Support more than just functions at the module level
    for (auto fun : ast->nodes)
      module->push_back(parseFunction(fun));
  }

  mlir::FuncOp Generator::parseProto(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Function && "Bad node");
    auto name = getFunctionName(ast);
    assert(!functionTable.inScope(name) && "Redeclaration");

    // Parse 'where' clause
    auto constraints = getFunctionConstraints(ast);
    for (auto c : constraints)
    {
      auto alias = getTokenValue(findNode(c, NodeType::ID));
      auto ty = findNode(c, NodeType::OfType);
      typeTable.insert(alias, parseType(ty.lock()));
    }

    // Function type from signature
    Types types;
    auto args = getFunctionArgs(ast);
    for (auto arg : args)
      types.push_back(parseType(getType(arg).lock()));
    auto retTy = parseType(getFunctionType(ast).lock());
    auto funcTy = builder.getFunctionType(types, retTy);

    // Create function
    auto func = mlir::FuncOp::create(getLocation(ast), name, funcTy);
    functionTable.insert(name, func);
    return func;
  }

  mlir::FuncOp Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Function && "Bad node");

    // Declare function signature
    TypeScopeT alias_scope(typeTable);
    auto name = getFunctionName(ast);
    if (!functionTable.inScope(name))
      parseProto(ast);
    auto func = functionTable.lookup(name);
    auto retTy = func.getType().getResult(0);

    // Create entry block
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto args = getFunctionArgs(ast);
    auto argVals = entryBlock.getArguments();
    assert(args.size() == argVals.size() && "Argument mismatch");
    for (auto var_val : llvm::zip(args, argVals))
    {
      llvm::StringRef name =
        findNode(std::get<0>(var_val).lock(), NodeType::ID).lock()->token;
      auto value = std::get<1>(var_val);
      declareVariable(name, value);
    }

    // Lower body
    auto body = getFunctionBody(ast);
    auto last = parseNode(body.lock());

    // Return last value
    if (last && last.getType() != retTy)
    {
      // Cast type (we trust the ast)
      last = genOperation(last.getLoc(), "verona.cast", {last}, retTy);
    }
    else
    {
      last = genOperation(getLocation(ast), "verona.none", {}, retTy);
    }
    builder.create<mlir::ReturnOp>(getLocation(ast), last);

    return func;
  }

  void Generator::declareVariable(llvm::StringRef name, mlir::Value val)
  {
    assert(!symbolTable.inScope(name) && "Redeclaration");
    symbolTable.insert(name, val);
  }

  void Generator::updateVariable(llvm::StringRef name, mlir::Value val)
  {
    assert(symbolTable.inScope(name) && "Variable not declared");
    symbolTable.update(name, val);
  }

  mlir::Value Generator::parseBlock(const ::ast::Ast& ast)
  {
    auto seq = findNode(ast, NodeType::Seq);
    mlir::Value last;
    for (auto sub : seq.lock()->nodes)
      last = parseNode(sub);
    return last;
  }

  mlir::Value Generator::parseNode(const ::ast::Ast& ast)
  {
    if (ast->is_token)
      return parseValue(ast);

    switch (ast->tag)
    {
      case NodeType::Localref:
        return parseValue(ast);
      case NodeType::Block:
        return parseBlock(ast);
      case NodeType::ID:
        return parseValue(ast);
      case NodeType::Assign:
        return parseAssign(ast);
      case NodeType::Call:
        return parseCall(ast);
      case NodeType::Let:
        return parseLet(ast);
      default:
        throw std::runtime_error("Node not implemented yet: " + ast->name);
    }
  }

  mlir::Value Generator::parseValue(const ::ast::Ast& ast)
  {
    assert(ast->is_token && "Bad node");

    // Variables
    if (ast->tag == NodeType::Localref)
    {
      auto var = symbolTable.lookup(ast->token);
      return var;
    }
    // TODO: Literals need attributes and types

    throw std::runtime_error("Value not implemented yet: " + ast->name);
  }

  mlir::Value Generator::parseAssign(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Assign && "Bad node");

    // The left-hand side must be an assignable value
    auto lhs = parseNode(ast->nodes[0]);

    // FIXME: For now this is only a variable and there are some redundancy
    // in the code.
    auto let = findNode(ast, NodeType::Let);
    llvm::StringRef name = findNode(let, NodeType::Local).lock()->token;

    // The right-hand side can be any expression
    // This is the value and we update the variable
    auto rhs = parseNode(ast->nodes[1]);
    updateVariable(name, rhs);
  }

  mlir::Value Generator::parseLet(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Let && "Bad node");
    assert(ast->nodes[0]->tag == NodeType::Local && "Bad node");
    llvm::StringRef name = findNode(ast, NodeType::Local).lock()->token;
    declareVariable(name, mlir::Value());
    return symbolTable.lookup(name);
  }

  mlir::Value Generator::parseCall(const ::ast::Ast& ast)
  {
    assert(ast->tag == NodeType::Call && "Bad node");
    llvm::StringRef name = findNode(ast, NodeType::Function).lock()->token;

    // All operations are calls, only calls to previously defined functions
    // are function calls. FIXME: Is this really what we want?
    if (functionTable.inScope(name))
    {
      // TODO: Lower calls
      return mlir::Value();
    }

    // Else, it should be an operation that we can lower natively
    // TODO: Separate between unary, binary, ternary, etc.
    // FIXME: Make this actually dynamic
    if (name == "+")
    {
      auto arg0 = parseNode(findNode(ast, NodeType::Localref).lock());
      auto arg1 = parseNode(findNode(ast, NodeType::Args).lock()->nodes[0]);
      auto dialect = Identifier::get("type", &context);
      auto type = mlir::OpaqueType::get(dialect, "ret", &context);
      return genOperation(getLocation(ast), "verona.add", {arg0, arg1}, type);
    }
  }

  mlir::Value Generator::genOperation(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> ops,
    mlir::Type retTy)
  {
    auto opName = OperationName(name, &context);
    auto state = OperationState(loc, opName);
    state.addOperands(ops);
    state.addTypes({retTy});
    auto op = builder.createOperation(state);
    return op->getResult(0);
  }

}
