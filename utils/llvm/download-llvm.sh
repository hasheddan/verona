#!/usr/bin/env bash

# Downloads LLVM build directory from Azure devops.

set -e

expected_args=4
if [[ $# != $expected_args ]]; then
    echo "Usage: $0 <arch> <os> <build> <hash>"
    echo "Example: $0 x86_64 linux release 3c123acf57c"
    exit 1
fi

arch=$1
os=$2
build=$3
commit=$4
image="verona-llvm-$arch-$os-$build-$commit"

if [ ! -f "/tmp/$image.tar.gz" ]; then
  az artifacts universal download \
    --organization "https://dev.azure.com/ProjectVeronaCI/" \
    --project "22b49111-ce1d-420e-8301-3fea815478ea" \
    --scope project \
    --feed "LLVMBuild" \
    --name "$image" \
    --version "*" \
    --path /tmp
else
  echo "Image already available in /tmp"
fi
