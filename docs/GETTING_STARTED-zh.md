# 入门指南：在 RUIBench 上测试智能体

[English](GETTING_STARTED.md) | 中文

本文说明如何安装基准、在上面运行你的智能体或模型，以及复现论文的实验。以下内容均已在全新检出的仓库上实际运行过。

## 1. 安装

需要 Python 3.10 或 3.11。核心部分（任务卡、Mock 引擎、评分、报告、RUIKit 工具库）几乎没有依赖：

```bash
git clone <本仓库> RUIBench && cd RUIBench
uv venv --python 3.10 .venv && source .venv/bin/activate   # 或：python3.10 -m venv .venv
uv pip install -e ".[dev]"                                  #     pip install -e ".[dev]"
ruibench validate          # 应为 0 个错误
ruibench run DS4 --seeds 0-1 --engine mock   # 参考策略跑两局，无需 GPU
```

（在 Ubuntu 上，`python3.10 -m venv` 需要 `python3.10-venv` 包；`uv` 不需要。）

正式结果来自 **Isaac Sim 4.5.0 + Isaac Lab 2.1**，需安装到独立的虚拟环境中：`ruibench doctor --hint` 会给出安装命令（需固定 `warp-lang==1.5.0`）。随后在该虚拟环境中执行：

```bash
pip install -e ".[isaac]"
export OMNI_KIT_ACCEPT_EULA=YES     # 阅读 NVIDIA 许可条款后设置
ruibench setup                      # 下载机器人、纹理和扫描物体
                                    # （约 2.6 GB，每个文件都固定了 sha256）
ruibench doctor                     # 检查运行环境
```

下载内容存放在 `third_party/`；设置 `RUIBENCH_THIRD_PARTY` 可让多个检出目录共用一份。新机器首次启动 Isaac 时要编译着色器，需要几分钟。

**在共享机器上：** Isaac 进程缺的是 CPU，不是 GPU：运行时约占 6 个核心（录像时 8–19 个），启动的一两分钟约占 40 个。各进程错开一分钟启动，整台机器的负载保持在核心数的 0.8 以下，开始批量运行前先告知大家（前两点 `ruikit run` 会自动做到）。`ruibench stop` 只停止当前检出目录启动的进程。CR6 例外：它的布料在显卡上解算（进程看到的第一块显卡，用 `CUDA_VISIBLE_DEVICES` 选择），显卡被别的任务占着时会慢好几倍。

## 2. 评测内容

- **30 项任务**，分为五类（世界知识、上下文学习、主动探索、动态交互、接触操作），每项有**简单**和**困难**两个难度，共 60 种任务与难度组合。`ruibench tasks` 列出全部任务；`ruibench show DS1` 显示一项。
- **种子。** 每项任务都是程序化生成的：种子决定一次任务的布置。`ruibench/tasks/splits.json` 把每种任务与难度组合的种子分为 **15 个开发种子（0–14）、20 个测试种子（15–34）、5 个备用种子**。在开发种子上开发（`--seeds dev`；`ruibench seeds DS1 easy` 会列出它们），在测试种子上报告（`--seeds test`）。论文的表格使用简单难度和每项任务的第一个测试种子（15）。
- **机器人。** 单臂任务：一台 Franka Research 3，配有桌对面的固定相机和一台能拍到手指的腕部相机。双臂任务（WK4、CL1、DI1、DI5、CR6）：两条 ARX X5 机械臂，配有一台头部相机，每条臂各有一台腕部相机。
- **规模。** 论文主表为 30 项任务 × 13 种 RUI × 1 个测试种子，即每个 Agent 390 局。在 Isaac 上，用真实模型跑一局约需 5–20 分钟。

## 3. 接入智能体

智能体只能通过 **RUIKit** 的工具操作机器人。13 种 RUI 使用论文中的名字——`joint`、`delta`、`step`、`pose`、`point`、`skill`，`program-joint` … `program-skill`，以及 `rui-free`——每个还有一个带版本的 id（`program-skill` 即 `rui.code-skill/v1`）；用名字运行时与论文一致，带辅助工具。`ruikit interfaces` 列出全部；详见 `docs/ruikit/README.md`。所有命令的选项写法一致：`--task`、`--level`、`--seeds`、`--interface`、`--agent`、`--model`、`--clock`、`--engine`、`--gpus`、`--out`。

Mock 引擎（`--engine mock`，无需 GPU）不渲染图像：模型在上面只能看到工具返回的内容，所以依赖外观的任务——例如 CL2 中长相相似、只能靠头发区分的玩偶——在上面无法完成，这类任务请在 Isaac 上开发。在 Isaac 上，所有接口都根据像素作答，因此要先启动视觉服务并设置 `RUIKIT_VISION_URL`（可用逗号分隔多个，分给各个工作进程；见 `docs/ruikit/perception.md`）。共有四种接入方式：

**通过 API 调用模型**，使用 RUIKit 自带的工具调用循环（`minimal-loop`）。通道定义在 `ruikit/configs/models.yaml`：`openai:<model>`（`OPENAI_API_KEY`；任何 OpenAI 兼容端点用 `OPENAI_BASE_URL`）、`anthropic:<model>`（`ANTHROPIC_API_KEY`）、`local:<model>`（vLLM 服务，`RUIKIT_LOCAL_BASE_URL`）。密钥从环境变量或 `.secrets/models.env`（已被 gitignore）读取。

```bash
ruikit episode --task DS1 --seeds dev:1 --interface pose --model openai:<model> --out runs/mine
ruikit run --agent minimal-loop --model openai:<model> --task WK6 WK5 --level easy \
    --seeds dev:3 --interface pose program-pose --out runs/mine      # --dry-run：只显示计划
```

`ruikit run` 按种子逐一运行 Agent × 任务 × 难度 × 种子 × 接口的组合，每局都在新的 Isaac 进程中运行；随时可以中断，再次运行同一命令即可继续。

**现成的编程 Agent**——Claude Code 或 Codex，各用自己的登录——每局一个全新的 Agent，只能通过 MCP 访问机器人：`--agent claude-code` 或 `--agent codex`（`codex-code`：Codex 自己的代码模式），并用 `--model` 指定模型。

**通过 stdio 的 MCP**，每个服务器一局，适用于任何支持 MCP 的 Agent：`ruikit serve-mcp --task DS2 --interface pose`。

**Python，进程内调用。** 每次调用都经过这一局本身：同一时钟、同一限制、同一日志。

```python
import ruikit
agent = ruikit.Agent(model="openai:<model>", interface="program-pose")
ruikit.run(agent, task="DS2", level="easy", seed=0, out="runs/mine")

from ruikit import Episode                       # 或者手动，一次调用一个工具
from ruikit.serve.sdk import Client
ep = Episode("DS2", level="easy", seed=0, profile="rui.pose/v1+aids", engine="isaac")
robot = Client(ep)
frame = robot.camera.look(cameras=["front"]).data["frames"][0]["frame_id"]
...
robot.task.done(report="the egg is in the basket")
print(ep.outcome)        # 是否成功，依据物理轨迹判定
ep.close()
```

**论文的实验**是一份实验清单，`ruikit/configs/experiments/paper.yaml`：

```bash
ruikit run --plan paper --dry-run                               # 列出全部要跑的局
ruikit run --plan paper --experiment rui --agent claude-code --model claude-opus-5-5
ruikit run --plan paper --experiment rui --interface program-delta program-step
ruikit report runs/paper                                        # 表格和结果页
```

## 4. 评分

一局是否成功由基准依据实际发生的物理过程判定，而不看智能体怎么说。每局都会写入一行 `ruibench-log/2` 格式的 JSON，包含完整轨迹：

```bash
ruikit rescore runs/mine --task DS1         # 仅凭日志为一次运行的各局重新评分；
                                            # 正式分数以此为准
ruibench view runs/mine/<agent>/<episode>/<episode>.episode.jsonl
                                            # 回放：俯视图、相机画面、时间轴
ruibench serve                              # 从其他机器访问 runs/（端口 8765）
```

`score.flags` 列出所有物理上不会发生的情况（瞬移、抛掷、关节超出限位）；被标记的局在有人复核之前判为失败。

## 5. 基本规则

- 智能体能获得的只有相机画面、机械臂读数和任务文本——不能获取仿真器状态、评分器、种子或物体的内部名称。任何工具都不得比真实机器人上的对应工具更精确。
- 不要在测试种子上调参。
- 任务卡、日志和视频都带有 canary 字符串（`ruibench/core/canary.py`）：不要让基准数据进入训练语料。

## 6. 版本

基准冻结为 **v1.0.0**，即论文所用版本：`T_ref` 为 8 秒，限时难度据此校准，种子划分已固定。`ruibench validate`（以及 `ruibench freeze --check`）会列出所有影响任务运行、且在冻结后改动过的文件；每个结果都记录其运行的版本，结果只在同一版本内可比。各版本的改动见 [CHANGELOG.md](../CHANGELOG.md)。
