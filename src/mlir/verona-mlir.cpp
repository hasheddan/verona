// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#include "CLI/CLI.hpp"
#include "ast/parser.h"
#include "ast/path.h"
#include "dialect/VeronaDialect.h"
#include "generator.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "llvm/Support/InitLLVM.h"

namespace
{
  enum class Source
  {
    NONE,
    VERONA,
    MLIR
  };
  struct Opt
  {
    std::string grammar;
    std::string filename;
    std::string output;
    bool mlir = false;
    bool llvm = false;
  };

  Opt parse(int argc, char** argv)
  {
    CLI::App app{"Verona MLIR"};

    Opt opt;
    app.add_flag("--emit-mlir", opt.mlir, "Emit MLIR.");
    app.add_flag("--emit-llvm", opt.llvm, "Emit LLVM (default).");
    app.add_option("-g,--grammar", opt.grammar, "Grammar to use.");
    app.add_option("-o,--output", opt.output, "Output filename.");
    app.add_option("file", opt.filename, "File to compile.");

    try
    {
      app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
      exit(app.exit(e));
    }

    // Default is to output MLIR
    if (!opt.llvm)
      opt.mlir = true;

    // Default grammar
    if (opt.grammar.empty())
      opt.grammar = path::directory(path::executable()).append("/grammar.peg");

    // Default input is stdin
    if (opt.filename.empty())
      opt.filename = "-";

    return opt;
  }
} // namespace

void help()
{
  std::cout << "Compiler Syntax: verona-mlir AST|MLIR|LLVM <filename.verona>"
            << std::endl;
}

Source getSourceType(llvm::StringRef filename)
{
  auto source = Source::NONE;
  if (filename.endswith(".verona"))
    source = Source::VERONA;
  else if (filename.endswith(".mlir"))
    source = Source::MLIR;
  else if (filename == "-") // STDIN, assume MLIR
    source = Source::MLIR;
  return source;
}

std::string getOutputFilename(llvm::StringRef filename, Opt& opt, Source source)
{
  if (!opt.output.empty())
    return opt.output;
  if (filename == "-")
    return "-";

  std::string newName = filename.substr(0, filename.find_last_of('.')).str();
  if (opt.mlir)
  {
    if (source == Source::MLIR)
      newName += ".mlir.out";
    else
      newName += ".mlir";
  }
  else
  {
    newName += ".ll";
  }
  return newName;
}

int main(int argc, char** argv)
{
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  // TODO: Register verona passes here.
  mlir::registerDialect<mlir::verona::VeronaDialect>();

  // Set up pretty-print signal handlers
  llvm::InitLLVM y(argc, argv);

  // MLIR Context
  mlir::MLIRContext context;

  // Parse cmd-line options
  auto opt = parse(argc, argv);
  llvm::StringRef filename(opt.filename);
  auto source = getSourceType(filename);
  if (source == Source::NONE)
  {
    std::cerr << "ERROR: Unknown source file " << filename.str()
              << ". Must be [verona, mlir]" << std::endl;
    return 1;
  }
  std::string outputFilename = getOutputFilename(filename, opt, source);

  // Generator
  mlir::verona::Generator gen;

  switch (source)
  {
    case Source::VERONA:
    {
      auto parser = parser::create(opt.grammar);
      auto ast = parser::parse(parser, opt.filename);
      if (!ast)
      {
        std::cerr << "ERROR: cannot parse Verona file " << filename.str()
                  << std::endl;
        return 1;
      }
      // Parse AST file into MLIR
      gen.readAST(ast);
      break;
    }
    case Source::MLIR:
      // Parse MLIR file
      gen.readMLIR(opt.filename);
      break;
    default:
      std::cerr << "ERROR: invalid source file type" << std::endl;
      return 1;
  }

  // Dump the MLIR graph
  if (opt.mlir)
  {
    auto mlir = gen.emitMLIR(outputFilename);
    if (!mlir)
    {
      std::cerr << "ERROR: failed to lower to MLIR" << std::endl;
      return 1;
    }
    return 0;
  }

  // Dump LLVM IR
  if (opt.llvm)
  {
    auto llvm = gen.emitLLVM(outputFilename);
    if (!llvm)
    {
      std::cerr << "ERROR: failed to lower to LLVM" << std::endl;
      return 1;
    }
    return 0;
  }

  return 0;
}
