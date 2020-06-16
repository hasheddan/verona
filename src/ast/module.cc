// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "module.h"

#include "path.h"

using namespace peg::udl;

namespace
{
  using namespace module;
  using Modules = std::map<std::string, ModulePtr>;

  ModulePtr make_module(const std::string& name)
  {
    return std::make_shared<Module>(name);
  }

  void extract(ast::Ast& ast, std::vector<std::string>& deps)
  {
    switch (ast->tag)
    {
      case "package_loc"_:
      {
        auto name = path::join(ast->path, ast->nodes.front()->token);

        if (path::extension(name).empty())
          name = path::to_directory(name);

        deps.push_back(name);
        return;
      }
    }

    ast::for_each(ast, extract, deps);
  }

  ModulePtr load(
    peg::parser& parser,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    Modules modules;
    std::vector<ModulePtr> stack;
    bool ok = true;

    auto canonical_path = path::canonical(path);
    auto m = make_module(path);
    modules.emplace(canonical_path, m);
    stack.push_back(m);

    while (!stack.empty())
    {
      m = stack.back();
      stack.pop_back();
      m->ast = parser::parse(parser, m->name, ext, err);

      if (!m->ast)
      {
        ok = false;
        continue;
      }

      std::vector<std::string> deps;
      extract(m->ast, deps);

      while (!deps.empty())
      {
        auto path = deps.back();
        deps.pop_back();

        canonical_path = path::canonical(path);
        auto find = modules.find(canonical_path);

        if (find != modules.end())
        {
          m->edges.push_back(find->second);
          continue;
        }

        auto dep = make_module(path);
        modules.emplace(canonical_path, dep);
        stack.push_back(dep);
        m->edges.push_back(dep);
      }
    }

    return modules.begin()->second;
  }

  bool run_passes(ModulePtr& m, const Passes& passes)
  {
    return dfs::post(
      m,
      [](auto& m, auto& passes) {
        bool ok = true;

        for (auto& pass : passes)
        {
          pass(m->ast, m->err);

          if (!m->err.empty())
          {
            ok = false;
            break;
          }
        }

        return ok;
      },
      passes);
  }

  bool gather_errors(ModulePtr& m, err::Errors& err)
  {
    return dfs::post(
      m,
      [](auto& m, auto& err) {
        err << m->err;
        return true;
      },
      err);
  }
}

namespace module
{
  ModulePtr build(
    peg::parser& parser,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto m = load(parser, path, ext, err);

    if (err.empty())
    {
      std::vector<std::pair<ModulePtr, ModulePtr>> pairs;
      dfs::cycles(
        m,
        [](auto& parent, auto& child, auto& pairs) {
          pairs.emplace_back(parent, child);
          return false;
        },
        pairs);

      for (auto& pair : pairs)
      {
        err << "These modules cause a cyclic dependency:" << std::endl
            << "  " << pair.second->name << std::endl
            << "  " << pair.first->name << err::end;
      }
    }

    if (err.empty())
    {
      if (!run_passes(m, passes))
        gather_errors(m, err);
    }

    return m;
  }

  ModulePtr build(
    const std::string& grammar,
    const Passes& passes,
    const std::string& path,
    const std::string& ext,
    err::Errors& err)
  {
    auto parser = parser::create(grammar, err);

    if (!err.empty())
      return {};

    return build(parser, passes, path, ext, err);
  }
}
