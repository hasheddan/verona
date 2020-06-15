#!/usr/bin/env bash

# Sets up the build dir by downloading the tar ball from Azure, unpacking on
# the external repository and changing the cmake paths to the current dir

git_root="$(git rev-parse --show-toplevel)"
if [ "$?" != "0" ]; then
  echo "Not in a git directory"
  exit 1
fi
utils_root="$git_root/utils/llvm"
llvm_root="$git_root/external/llvm-project"
devops_root="/agent/_work/1/s"
arch="x86_64"
os="linux"
build="release"
commit="3c123acf57c"

# Download build cache to /tmp
echo "Downloading llvm's build cache"
bash "$utils_root/download-llvm.sh" $arch $os $build $commit

# Unpack into llvm's directory
echo "Rebuilding LLVM's build directory"
rm -rf "$llvm_root/build"
tar zxf /tmp/verona-llvm-$arch-$os-$build-$commit.tar.gz --directory "$llvm_root"

# Change LLVMConfig with current path
echo "Changing CMake's paths"
for file in $(find "$llvm_root/build" -name \*.cmake); do
  sed -ri "s,$devops_root,$llvm_root,g" "$file"
done
