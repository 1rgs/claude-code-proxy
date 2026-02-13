#!/usr/bin/env bash
#
# Claude Code Proxy — create Python virtual environment and check prerequisites.
# If venv already exists, it is removed and recreated from scratch.
# Prerequisites: curl, git, Homebrew (macOS), Python 3.10+, uv, Google Cloud SDK (gcloud).
# Run from repo root.
#
set -e

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Detect OS
case "$(uname -s)" in
    Darwin)  OS="macos" ;;
    Linux)   OS="linux" ;;
    *)       OS="other" ;;
esac

echo "=== Claude Code Proxy — env setup (prerequisites + venv) ==="
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# curl — required for downloading installers (Homebrew, uv, gcloud)
# -----------------------------------------------------------------------------
if ! command -v curl &>/dev/null; then
    if [[ "$OS" == "macos" ]] && command -v xcode-select &>/dev/null; then
        echo "Installing Xcode Command Line Tools (provides curl)..."
        xcode-select --install 2>/dev/null || true
        echo "After the installer finishes, re-run this script."
        exit 1
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            echo "Installing curl..."
            sudo apt-get update -qq && sudo apt-get install -y curl
        elif command -v dnf &>/dev/null; then
            echo "Installing curl..."
            sudo dnf install -y curl
        else
            echo "Error: curl is required. Install curl and re-run."
            exit 1
        fi
    else
        echo "Error: curl is required for installs. Install curl and re-run."
        exit 1
    fi
fi
echo "curl: $(curl --version 2>/dev/null | head -1 || true)"

# -----------------------------------------------------------------------------
# Homebrew (macOS) — used to install Python, git, and gcloud if missing
# -----------------------------------------------------------------------------
if [[ "$OS" == "macos" ]]; then
    if ! command -v brew &>/dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for this session (common post-install path on Apple Silicon)
        if [[ -x /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -x /usr/local/bin/brew ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    echo "Homebrew: $(brew --version 2>/dev/null | head -1 || true)"
fi

# -----------------------------------------------------------------------------
# git — optional; used for cloning and version control
# -----------------------------------------------------------------------------
if ! command -v git &>/dev/null; then
    if [[ "$OS" == "macos" ]] && command -v brew &>/dev/null; then
        echo "Installing git via Homebrew..."
        brew install git
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            echo "Installing git..."
            sudo apt-get install -y git
        elif command -v dnf &>/dev/null; then
            echo "Installing git..."
            sudo dnf install -y git
        fi
    fi
fi
if command -v git &>/dev/null; then
    echo "git: $(git --version 2>/dev/null || true)"
    # Check if git credentials look configured (for cloning private repos)
    GIT_CRED_OK=false
    if git config --global credential.helper &>/dev/null; then
        GIT_CRED_OK=true
    fi
    if [[ -f "${HOME}/.ssh/id_ed25519" ]] || [[ -f "${HOME}/.ssh/id_rsa" ]]; then
        GIT_CRED_OK=true
    fi
    if [[ "$GIT_CRED_OK" != true ]]; then
        echo "  Note: No git credential helper or SSH key found. For private repos, set up: git config credential.helper store (or cache), or add an SSH key to your Git host."
    fi
fi

# -----------------------------------------------------------------------------
# Python 3.10+ (required)
# -----------------------------------------------------------------------------
PYTHON_BIN=""
PIP_BIN=""

if command -v pyenv &>/dev/null; then
    eval "$(pyenv init -)" 2>/dev/null || true
    if ! pyenv versions --bare 2>/dev/null | grep -qE '^3\.(10|11|12)'; then
        echo "Installing Python 3.11.11 via pyenv..."
        pyenv install 3.11.11
    fi
    pyenv local 3.11.11 2>/dev/null || true
    PYTHON_BIN="python"
    PIP_BIN="pip"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    for v in 3.11 3.10 3.12; do
        if command -v "python${v}" &>/dev/null; then
            PYTHON_BIN="python${v}"
            PIP_BIN="pip${v}"
            break
        fi
    done
    if [[ -z "$PYTHON_BIN" ]] && command -v python3 &>/dev/null; then
        ver=$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "0")
        if [[ -n "$ver" ]] && [[ "$ver" -ge 10 ]]; then
            PYTHON_BIN="python3"
            PIP_BIN="pip3"
        fi
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ "$OS" == "macos" ]] && command -v brew &>/dev/null; then
        echo "Installing Python 3.11 via Homebrew..."
        brew install python@3.11
        PYTHON_BIN="python3.11"
        PIP_BIN="pip3.11"
    else
        echo "Error: Python 3.10+ not found. Install it (e.g. pyenv, brew install python@3.11, or system package)."
        exit 1
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: Could not resolve Python. Install Python 3.10+ and re-run."
    exit 1
fi

echo "Python: $($PYTHON_BIN --version)"

# -----------------------------------------------------------------------------
# Google Cloud SDK (gcloud) — optional for running proxy (needed for Vertex scripts)
# -----------------------------------------------------------------------------
if ! command -v gcloud &>/dev/null; then
    if [[ "$OS" == "macos" ]] && command -v brew &>/dev/null; then
        echo "Installing Google Cloud SDK (gcloud) via Homebrew..."
        brew install --cask google-cloud-sdk
        for prefix in /opt/homebrew /usr/local; do
            gcp="$prefix/Caskroom/google-cloud-sdk/latest/google-cloud-sdk"
            if [[ -d "$gcp" ]]; then
                export PATH="$gcp/bin:$PATH"
                break
            fi
        done
    elif [[ "$OS" == "linux" ]]; then
        echo "Installing Google Cloud SDK (gcloud)..."
        GCLOUD_ROOT="${HOME}/google-cloud-sdk"
        curl -fsSL https://sdk.cloud.google.com | bash -s -- --install-dir="${HOME}" --disable-prompts
        if [[ -f "${GCLOUD_ROOT}/path.bash.inc" ]]; then
            # shellcheck source=/dev/null
            source "${GCLOUD_ROOT}/path.bash.inc"
        fi
        export PATH="${GCLOUD_ROOT}/bin:${PATH}"
    else
        echo "Warning: gcloud not found. Install from https://cloud.google.com/sdk/docs/install for Vertex scripts. Proxy can still run with a service account key file."
    fi
fi

if command -v gcloud &>/dev/null; then
    echo "gcloud: $(gcloud --version 2>/dev/null | head -1 || true)"
fi

# -----------------------------------------------------------------------------
# uv — Python package manager/runner (used in README/Dockerfile: uv run ...)
# -----------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
fi

if command -v uv &>/dev/null; then
    echo "uv: $(uv --version 2>/dev/null || true)"
fi

# -----------------------------------------------------------------------------
# Virtual environment and project install (using uv)
# -----------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    echo "Removing existing $VENV_DIR (starting over)..."
    rm -rf "$VENV_DIR"
fi

if command -v uv &>/dev/null; then
    echo "Creating virtual environment in $VENV_DIR with uv..."
    uv venv "$VENV_DIR" --python "$PYTHON_BIN"
    echo "Installing project and dependencies with uv (from lockfile)..."
    if [[ -f "$REPO_ROOT/uv.lock" ]]; then
        uv sync --locked
    else
        uv sync
    fi
else
    echo "Creating virtual environment in $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Installing project and dependencies with pip..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -e .
fi

echo ""
echo "=== Done ==="
echo "  Activate:   source $VENV_DIR/bin/activate"
echo "  Run proxy:  uv run uvicorn server:app --host 0.0.0.0 --port 8082"
echo "              (or with env active: uvicorn server:app --host 0.0.0.0 --port 8082)"
echo "  .env:       cp .env.example .env  then edit .env"
if ! command -v gcloud &>/dev/null; then
    echo "  (Install gcloud for Vertex: https://cloud.google.com/sdk/docs/install)"
fi
