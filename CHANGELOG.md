# Changelog

## v1.0.0 -- the paper's version

The first public release: the benchmark and RUIKit as the paper's experiments ran them.

- **The benchmark.** 30 tasks in five suites (World knowledge, In-context learning,
  Discovery, Dynamic interaction, Contact-rich manipulation), each at an easy and a hard
  level; 15 development and 20 test missions per task and level; `T_ref` = 8 s. The task
  cards, their code, the scoring, the engines, the interfaces and the harnesses are
  fingerprinted in `ruibench/tasks/frozen.json`; `ruibench validate` names any file that
  differs.
- **RUIKit.** Six action abstractions in two interaction forms -- 12 fixed RUIs -- and
  RUI-Free: 13 interfaces, named as in the paper (`skill`, `program-skill`, `rui-free`)
  over versioned ids (`rui.skill/v1`, `rui.code-skill/v1`, `rui.free/v1`); the
  action-sequence variant and the action-level checks of the paper's ablation.
- **What an agent is given is the paper's.** Every interface's tools, as each provider and
  MCP are sent them, and the prompts of both kinds of agent are those of the paper's runs
  (`tests/ruikit/test_protocol_unchanged.py`).
- **One way to run it.** `ruibench` (tasks, engines, scoring, reports) and `ruikit` (agents)
  spell their options the same way: `--task`, `--level`, `--seeds`, `--interface`,
  `--agent`, `--model`, `--clock`, `--engine`, `--gpus`, `--out`. The paper's experiments
  are a plan: `ruikit run --plan paper`.
- **Models.** The minimal tool-calling loop reaches models through OpenAI, Anthropic, a
  local vLLM server or QwenCloud (`ruikit/configs/models.yaml`); Claude Code and Codex run
  as themselves, one fresh agent an episode over MCP.
