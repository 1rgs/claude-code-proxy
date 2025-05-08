#!/bin/bash
[ "$1" = -x ] && shift && set -x
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

VENV_DIR="venv"
PYTHON_CMD="python3"
MIN_PYTHON_VERSION="3.10.0"

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   Claude Code Proxy Setup Script     ${NC}"
echo -e "${BLUE}=======================================${NC}"

command_exists() {
  command -v "$1" &> /dev/null
}

check_python_version() {
  local python_version
  python_version=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
  local required_version=$MIN_PYTHON_VERSION

  echo -e "${YELLOW}Checking Python version: ${python_version}${NC}"

  if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo -e "${YELLOW}Python version ${python_version} is lower than required version ${required_version}${NC}"
    return 1
  fi
  return 0
}

ensure_uv_installed() {
  if ! command_exists uv; then
    echo -e "${YELLOW}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [[ -f "$HOME/.cargo/env" ]]; then
      source "$HOME/.cargo/env"
    fi

    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo -e "${GREEN}uv is already installed!${NC}"
  fi
}

setup_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Creating virtual environment in ./$VENV_DIR${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
  else
    echo -e "${GREEN}Virtual environment already exists!${NC}"
  fi

  echo -e "${YELLOW}Activating virtual environment...${NC}"
  source "$VENV_DIR/bin/activate"

  if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}Virtual environment activated: $VIRTUAL_ENV${NC}"
  else
    echo -e "${YELLOW}Failed to activate virtual environment.${NC}"
    exit 1
  fi
}

install_dependencies() {
  echo -e "${YELLOW}Installing dependencies using uv...${NC}"
  uv pip install -e .
  echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

setup_env_file() {
  if [[ ! -f ".env.example" ]]; then
    echo -e "${YELLOW}ERROR: .env.example file is missing!${NC}"
    echo -e "${YELLOW}Please ensure .env.example exists in the project root.${NC}"
    exit 1
  fi

  if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}Creating .env file from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}.env file created! Please update it with your API keys.${NC}"
  else
    echo -e "${GREEN}.env file already exists!${NC}"
  fi
}

main() {
  if ! check_python_version; then
    echo -e "${YELLOW}Please install Python $MIN_PYTHON_VERSION or higher!${NC}"
    exit 1
  fi

  ensure_uv_installed
  setup_venv
  install_dependencies
  setup_env_file

  echo -e "${GREEN}Setup complete!${NC}"
  echo -e "${BLUE}To activate the virtual environment in the future, run:${NC}"
  echo -e "    source $VENV_DIR/bin/activate"
  echo -e "${BLUE}To run the server:${NC}"
  echo -e "    python -m uvicorn server:app --host 0.0.0.0 --port 8082 --reload"
  echo -e "${BLUE}To use with Claude Code:${NC}"
  echo -e "    ANTHROPIC_BASE_URL=http://localhost:8082 claude"
}

main
