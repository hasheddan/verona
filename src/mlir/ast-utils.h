// Copyright (c) Microsoft Corporation. All rights reserved.
// This file is licensed under the MIT license.

#include "ast/cli.h"
#include "ast/files.h"
#include "ast/parser.h"
#include "ast/path.h"
#include "ast/sym.h"

#include <string>
#include <vector>

// This is a bag of utility functions to handle AST lookups and fail-safe
// operation. While the AST design is still in flux, we can keep this around,
// but once we're set on its structure, this should move to src/ast instead.

namespace mlir::verona {
  // We need this becayse the ""_ operator doens't stack well outside of the
  // peg namespace, so we need to call str2tag directly. Easier to do so in a
  // constexpr enum type creation and let the rest be unsigned comparisons.
  // The AST code needs to be flexible, so using the operator directly is more
  // convenient. But we need to be very strict (with MLIR generation), so this
  // also creates an additional layer of security.
  enum class NodeType : unsigned int {
    None = 0,
    Module = peg::str2tag("module"),
    Function = peg::str2tag("function"),
    FuncName = peg::str2tag("funcname"),
    Sig = peg::str2tag("sig"),
    Block = peg::str2tag("block"),
    OfType = peg::str2tag("oftype"),
    Type = peg::str2tag("type"),
    TypeRef = peg::str2tag("type_ref"),
    Params = peg::str2tag("params"),
    NamedParam = peg::str2tag("namedparam"),
    // TODO: Add all
  };
  const static bool operator==(const unsigned int tag, const NodeType type) {
    return tag == (unsigned int)type;
  }

  // Find a sub-node of tag 'type'
  ::ast::WeakAst findNode(::ast::WeakAst ast, NodeType type);

  // Get token value or return "unk{inc++}" if empty
  const std::string getTokenValue(::ast::WeakAst ast);

  // Type helpers
  ::ast::WeakAst getType(::ast::WeakAst ast);
  const std::string getTypeDesc(::ast::WeakAst ast);

  // Function helpers
  const std::string getFunctionName(::ast::WeakAst ast);
  ::ast::WeakAst getFunctionType(::ast::WeakAst ast);
  std::vector<::ast::WeakAst> getFunctionArgs(::ast::WeakAst ast);
  ::ast::WeakAst getFunctionBody(::ast::WeakAst ast);
}
