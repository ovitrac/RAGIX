# Initializa environment variables for RAGIX

# Which Ollama model to use
export UNIX_RAG_MODEL="mistral"          # or mistral:latest, qwen2.5, etc.

# Sandbox root for all shell operations
export UNIX_RAG_SANDBOX="$HOME/projects" # or any directory you want

# Agent profile / safety:
#   safe-read-only : only non-destructive commands
#   dev            : dev-friendly defaults
#   unsafe         : allow everything
export UNIX_RAG_PROFILE="dev"

# Allow destructive git operations (reset, clean, etc.)
export UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=0  # set to 1 if you really want this

