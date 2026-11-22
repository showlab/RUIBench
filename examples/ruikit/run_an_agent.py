"""A Robot Agent -- a model, a Harness, an interface and its components -- run on
one task under two conditions.

    python examples/ruikit/run_an_agent.py openai:<model>

The model is reached through a channel in ruikit/configs/models.yaml; its key comes
from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY) or .secrets/models.env.
Each run writes one cell of the results layout under runs/examples and prints its row.
"""

import json
import sys

import ruikit

if len(sys.argv) < 2:
    sys.exit("which model? e.g. openai:<model>, anthropic:<model>, local:<served name>")
agent = ruikit.Agent(model=sys.argv[1], harness="minimal-loop",
                     interface="program-pose")    # rui.code/v1, with the aids as the paper ran it
print("interface:", agent.interface_id())

for condition in ("8s", "paused"):
    row = ruikit.run(agent, task="WK1", level="easy", seed=0, condition=condition,
                     engine="mock", out="runs/examples")
    print(condition, json.dumps({k: row.get(k) for k in ("success", "reason", "replies",
                                                          "sim_seconds")}))
