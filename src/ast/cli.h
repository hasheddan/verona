// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

namespace cli
{
  struct Opt
  {
    bool ast = false;
    std::string grammar;
    std::string filename;
  };

  Opt parse(int argc, char** argv);
}
