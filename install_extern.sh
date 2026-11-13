#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Installing numeric-type-quad extern..."
cd "$SCRIPT_DIR/extern/numeric-type-quad"
bash install_extern.sh

echo ""
echo "==> Done! All extern dependencies ready."
echo ""
echo "    eigen:          $SCRIPT_DIR/extern/eigen"
echo "    nlohmann_json:  $SCRIPT_DIR/extern/nlohmann_json"
echo "    spdlog:         $SCRIPT_DIR/extern/spdlog"
echo "    numeric-type-quad: $SCRIPT_DIR/extern/numeric-type-quad"
echo ""
echo "Now run: bash compile.sh"
