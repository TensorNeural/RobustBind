#!/bin/bash

# Start tmux session and run btop & nvtop together
tmux new-session -d -s monitoring "btop"
tmux split-window -h "nvtop"
tmux -2 attach-session -d
