"""A tool from outside the library: registered when this file is imported."""

from dataclasses import dataclass

from ruikit import ToolResult, param, tool


@dataclass(frozen=True)
class EchoParams:
    text: str = param("what to say back")


@tool(name="debug_echo", family="compute", params=EchoParams, source="agent",
      examples=[{"text": "hello"}],
      description="""Say the given text back, unchanged. For checking that a tool from outside
      the library reaches the agent: it touches nothing, reads no camera and costs no
      simulated time.""")
def debug_echo(ctx, p: EchoParams) -> ToolResult:
    return ToolResult(data={"echo": p.text}, text=p.text)
