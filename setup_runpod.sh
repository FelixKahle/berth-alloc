#!/usr/bin/env bash
# setup_rust_runpod_clean.sh
# Sets up Rust in a plain Runpod Ubuntu container.
# - Updates packages; installs base build tools
# - Ensures git (if missing), clang (if missing), cmake (if missing)
# - Installs rustup (non-interactive), rustc, cargo, rustfmt, clippy
# - Verifies installation
# Idempotent: safe to run multiple times.

set -euo pipefail

# Use sudo only if not running as root
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "==> Updating package lists & upgrading..."
$SUDO apt-get update -y
$SUDO apt-get upgrade -y

# Base build tools (installed regardless; harmless if already present)
BASE_PKGS=(ca-certificates curl build-essential pkg-config libssl-dev)

# Conditionally add tools only if missing
EXTRA_PKGS=()
if ! command -v git >/dev/null 2>&1;   then EXTRA_PKGS+=(git);   fi
if ! command -v clang >/dev/null 2>&1; then EXTRA_PKGS+=(clang); fi
if ! command -v cmake >/dev/null 2>&1; then EXTRA_PKGS+=(cmake); fi

echo "==> Installing required packages..."
$SUDO apt-get install -y --no-install-recommends "${BASE_PKGS[@]}" "${EXTRA_PKGS[@]}"
$SUDO apt-get autoremove -y
$SUDO apt-get clean

# Install rustup if missing
if ! command -v rustup >/dev/null 2>&1; then
  echo "==> Installing rustup (non-interactive)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
  echo "==> rustup already present — skipping install."
fi

# Activate PATH for current shell session (no shell rc writes)
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

# Ensure ~/.cargo/bin is permanently on PATH (idempotent)
if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
fi
if [[ -f "$HOME/.profile" ]] && ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.profile"; then
  echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.profile"
fi

# Ensure stable toolchain and components
echo "==> Ensuring stable toolchain + components..."
rustup toolchain install stable -q || true
rustup default stable -q || true
rustup component add rustfmt clippy -q || true

# Verify installation
echo "==> Verifying installation..."
set +e
rustc --version
cargo --version
rustup --version
clang --version 2>/dev/null | head -n1
cmake --version 2>/dev/null | head -n1
git --version 2>/dev/null | head -n1
set -e

echo "==> Done. Rust is installed and configured."
echo "    For new shells: source \"$HOME/.cargo/env\""
