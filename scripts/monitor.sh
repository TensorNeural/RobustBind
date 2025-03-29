#!/bin/bash

SESSION_NAME="monitoring"

# Check if the session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  # Attach to the existing session
  tmux attach-session -t $SESSION_NAME
else
  # Start a new session with btop
  tmux new-session -d -s $SESSION_NAME "btop"
  # Split and run nvtop
  tmux split-window -h "nvtop"
  # Attach to the session
  tmux -2 attach-session -t $SESSION_NAME
fi
