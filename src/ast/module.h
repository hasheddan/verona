// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ast.h"
#include "dfs.h"
#include "err.h"
#include "parser.h"

namespace module
{
  using Pass = void (*)(ast::Ast& ast, err::Errors& err);
  using Passes = std::vector<Pass>;

  struct Module
  {
    std::string name;
    ast::Ast ast;
    err::Errors err;

    dfs::Color color;
    std::vector<std::shared_ptr<Module>> edges;

    Module(const std::string& name) : name(name), color(dfs::white) {}
  };

  using ModulePtr = std::shared_ptr<Module>;

  ModulePtr build(
    peg::parser& parser,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err);

  ModulePtr build(
    const std::string& grammar,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err);

  template<typename T>
  T& operator<<(T& out, ModulePtr m)
  {
    struct Stream : public dfs::Default<Module>
    {
      T& out;
      Stream(T& out) : out(out) {}

      bool post(ModulePtr& m)
      {
        out << peg::ast_to_s(m->ast);
        return true;
      }
    };

    Stream s(out);
    dfs::dfs(m, s);
    return out;
  }
}
