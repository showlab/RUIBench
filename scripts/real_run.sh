#!/usr/bin/env bash
#
#   scripts/real_run.sh DS2 --test                   # first, a try: the pose alone, which counts for nothing
#   scripts/real_run.sh DS2                          # Claude Code + Opus 5.5
#   scripts/real_run.sh DS2 --agent codex            # Codex + GPT-6 Astra
#   scripts/real_run.sh DS2 --seed 16                # another table: another seed of its card
#   scripts/real_run.sh R0                           # the demo: the lemon on the plate
#
set -eu
cd "$(dirname "$0")/.."

# Real-Robot tasks: DS2, CL3, DI4
TASK=${1:?which task? a card of ruibench/tasks/real/tasks.yaml, as DS2}
shift

# In the order they run: the pose first, the moves the calibration's check has seen
INTERFACES=(
    # one action a reply
    pose
    point
    skill
    step
    delta
    joint
    # the same six, as programs
    program-pose
    program-point
    program-skill
    program-step
    program-delta
    program-joint
    # RUI-Free
    rui-free
)

PYTHON=.venv/bin/python
VISION=http://127.0.0.1:18300/info

# The station answers (nothing moves)
"$PYTHON" -m ruibench.engines.franka.check

# The vision service, if it is not up; it stays up for the next run
if ! curl -sf -m 2 "$VISION" > /dev/null; then
    echo "starting the vision service (SAM 3) ..."
    mkdir -p runs/real
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. setsid .venv-vision/bin/python \
        -m ruikit.perception.service.server > runs/real/vision-server.log 2>&1 < /dev/null &
    until curl -sf -m 2 "$VISION" > /dev/null; do
        kill -0 $! 2> /dev/null || { tail -5 runs/real/vision-server.log; exit 1; }
        sleep 2
    done
fi

# Once on this robot: small moves, low over the table, and how far the arm runs on after P
"$PYTHON" -m ruibench.engines.franka.calibrate moves --once

exec "$PYTHON" -m ruikit.experiments.real --task "$TASK" --agent claude-code \
    --interface "${INTERFACES[@]}" "$@"
