"""From a tool log to the decision cards of a page: what each call asked for and what it came to."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ruikit.report.labels import (
    SIZE_WORDS,
    STEP_WORDS,
)


# --------------------------------------------------------------------------
# a clip's decisions
# --------------------------------------------------------------------------
def _num(x, nd=3):
    if not isinstance(x, (int, float)):
        return str(x)
    text = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _vec(v, nd=3):
    return "[" + ", ".join(_num(x, nd) for x in v) + "]"


def _outcome(result: dict) -> (str, bool):
    """A reply's gist, and whether it went as asked."""
    if not result.get("ok", True):
        return f"{result.get('error', 'error')}: {result.get('text', '')}".strip(": "), False
    data = result.get("data") or {}
    status = data.get("status")
    bits = []
    if status:
        bits.append({"done": "完成", "failed": "未到位", "running": "仍在动",
                     "cancelled": "已停"}.get(status, status))
    if data.get("residual_cm") is not None and status != "done":
        bits.append(f"差 {data['residual_cm']} cm")
    if "holding_something" in data and status:
        bits.append("夹着东西" if data["holding_something"] else "手里空")
    if "tool_calls" in data:
        bits.append(f"程序里调用了 {data['tool_calls']} 次工具")
    if data.get("error"):
        bits.append(str(data["error"]))
    return " · ".join(bits), status in (None, "done")


def _call(tool: str, args: dict, result: dict, clip_url: str, shown: Dict[str, dict],
          two_arms: bool) -> dict:
    """One tool call as a card."""
    arm = args.get("arm")
    who = f"{'左手' if arm == 'left' else '右手'}：" if two_arms and arm else ""
    card = {"kind": "other", "text": f"{tool}", "tool": tool}
    if tool == "robot_step":
        move, size = args.get("move", ""), args.get("size", "fine")
        words = STEP_WORDS.get(move, move)
        if move not in ("grasp", "release"):
            words += f" · {SIZE_WORDS.get(size, size)}"
        card.update(kind="step", text=who + words, code=f'{move}' + ("" if move in ("grasp", "release")
                                                                       else f' {size}'))
    elif tool == "robot_move_to":
        rpy = args.get("rpy_deg")
        turn = f"  朝向 {_vec(rpy, 1)}°" if rpy and any(abs(x) > 0.05 for x in rpy) else ""
        card.update(kind="move", text=f"{who}到 {_vec(args.get('xyz_m', []))} m{turn}")
    elif tool == "robot_move_to_pixel":
        f = args.get("frame_id", "")
        pic = shown.get(f)
        h = float(args.get("height_m", 0) or 0)
        if args.get("mode", "surface") == "surface":
            where = f"比那里的表面{'高' if h >= 0 else '低'} {_num(abs(h))} m"
        else:
            where = f"在桌面以上 {_num(h)} m 处"
        card.update(kind="point", text=f"{who}指 {f} 上的像素 ({_num(args.get('u'), 1)}, "
                                       f"{_num(args.get('v'), 1)})，{where}")
        if pic:
            card["thumbs"] = [{**pic, "u": args.get("u"), "v": args.get("v")}]
    elif tool == "camera_look":
        frames = (result.get("data") or {}).get("frames") or []
        card.update(kind="look", text="看：" + "、".join(fr.get("camera", "") for fr in frames))
        card["thumbs"] = [shown[fr["frame_id"]] for fr in frames if fr.get("frame_id") in shown]
    elif tool == "vision_detect":
        objs = (result.get("data") or {}).get("objects") or []
        q = args.get("query") or "全部"
        card.update(kind="detect", text=f"在 {args.get('frame_id', '')} 里找“{q}”：{len(objs)} 个")
        pic = shown.get(args.get("frame_id", ""))
        if pic:
            card["thumbs"] = [{**pic, "boxes": [list(o["box"]) + [o["id"].split("#")[-1]]
                                                for o in objs if o.get("box")]}]
    elif tool == "vision_pose":
        d = result.get("data") or {}
        card.update(kind="pose", text=f"估 {args.get('id', '')}（{d.get('label', '')}）的位姿："
                                      f"位置 {_vec(d.get('xyz_m', []))} m，偏航 {_num(d.get('yaw_deg'), 1)}°")
    elif tool == "vision_lift":
        d = result.get("data") or {}
        card.update(kind="lift", text=f"把 {args.get('frame_id', '')} 的像素 ({_num(args.get('u'), 1)}, "
                                      f"{_num(args.get('v'), 1)}) 变成三维点：{_vec(d.get('xyz_m', []))} m"
                                      + (f"（在 {d['on']} 上）" if d.get("on") else ""))
    elif tool == "robot_state":
        arms = (result.get("data") or {}).get("arms") or []
        bits = [f"{'左手' if a.get('arm') == 'left' else ('右手' if two_arms else '手')}在 "
                f"{_vec(a['tcp']['xyz_m'])}" for a in arms if a.get("tcp")]
        card.update(kind="state", text="读自己的状态：" + "；".join(bits))
    elif tool == "robot_gripper":
        what = args.get("command") or f"张开到 {_num(args.get('width_m'))} m"
        card.update(kind="grip", text=who + {"close": "夹紧", "open": "松开"}.get(what, what))
    elif tool == "robot_wait":
        if args.get("seconds") is not None:
            card.update(kind="wait", text=f"等 {_num(args['seconds'], 2)} s")
        elif args.get("until_event"):
            card.update(kind="wait", text="等下一件事发生")
        else:
            card.update(kind="wait", text="等动作做完")
    elif tool == "robot_stop":
        card.update(kind="stop", text=who + "停")
    elif tool == "code_run":
        src = args.get("source", "")
        card.update(kind="code", text=f"写了一段 {len(src.splitlines())} 行的程序", code=src)
        data = result.get("data") or {}
        found = data.get("result")
        if isinstance(found, dict) and isinstance(found.get("found"), dict):
            found = found["found"]                 # a program that also returns its frames
        shaped = [(k, v) for k, v in (found.items() if isinstance(found, dict) else ())
                  if isinstance(v, list) and len(v) >= 2 and isinstance(v[0], str)
                  and isinstance(v[1], (list, tuple))]
        if "vision.pose" in src and shaped:        # the grounded agent's look-and-estimate
            card["found"] = [f"{k}（{v[0]}）{_vec(v[1])} m" for k, v in shaped[:8]]
        elif found is not None or data.get("stdout"):
            # anyone else's program: what it returned and printed, as it came
            shown = json.dumps(found, ensure_ascii=False, default=str) if found is not None else ""
            printed = (data.get("stdout") or "").strip()
            text = "；".join(x for x in (shown and f"result {shown}", printed and f"print {printed}") if x)
            card["found"] = [text[:600] + ("…" if len(text) > 600 else "")]
    elif tool == "task_done":
        card.update(kind="done", text=f"报告完成：{args.get('report', '')}")
    card["result"], card["ok"] = _outcome(result)
    return card


def _pictures(result: dict, clip_url: str) -> Dict[str, dict]:
    out = {}
    for im in result.get("images") or ():
        if im.get("path"):
            out[im["frame_id"]] = {"src": f"{clip_url}/images/{im['path']}",
                                   "w": im.get("width"), "h": im.get("height"),
                                   "camera": im.get("camera")}
    return out


def _inner(calls: List[dict]) -> str:
    """The tools a program called, in order, a run of one tool counted."""
    out: List[List] = []
    for c in calls:
        name = c["tool"].replace("_", ".", 1)
        if out and out[-1][0] == name:
            out[-1][1] += 1
        else:
            out.append([name, 1])
    return " → ".join(f"{n} ×{k}" if k > 1 else n for n, k in out)


def decisions(rows: List[dict], clip_url: str, two_arms: bool,
              notes: Optional[Dict[tuple, str]] = None) -> List[dict]:
    """Every reply the interface sent, as cards: a run of the same step is one
    card with a count, and a look straight before a pointing is shown on it.
    ``notes``: (row, call) -> where that call's coordinates came from."""
    notes = notes or {}
    shown: Dict[str, dict] = {}
    cards: List[dict] = []
    for row in rows:
        calls = (row["args"].get("calls") or []) if row["tool"] == "batch" else \
            [{"tool": row["tool"], "args": row.get("args") or {}}]
        results = row.get("results") or []
        for res in results:
            shown.update(_pictures(res, clip_url))
        parts = [_call(c["tool"], c.get("args") or {}, results[k] if k < len(results) else {},
                       clip_url, shown, two_arms) for k, c in enumerate(calls)]
        if not parts:
            continue
        for k, p in enumerate(parts):
            if (row.get("i"), k) in notes:
                p["note"] = notes[(row.get("i"), k)]
        if row.get("inner"):
            parts[0]["inner"] = _inner(row["inner"])
        card = parts[0] if len(parts) == 1 else _merge(parts, calls)
        card.update(t0=row["t0"], t1=row["t1"], n=1, turn=row.get("turn"))
        prev = cards[-1] if cards else None
        if (prev and card["kind"] in ("step", "wait", "code") and prev["kind"] == card["kind"]
                and prev["text"] == card["text"] and prev.get("code") == card.get("code")
                and prev["ok"] == card["ok"] and not card.get("note")):
            # the same step, the same wait, the same program again: one card, counted
            prev["n"] += 1
            prev["t1"] = card["t1"]
            continue
        if prev and card["kind"] == "point" and prev["kind"] == "look":
            chosen = {t["src"] for t in card.get("thumbs", [])}
            card["saw"] = [t for t in prev.get("thumbs", []) if t["src"] not in chosen]
            card["t0"] = prev["t0"]
            cards[-1] = card
            continue
        cards.append(card)
    return cards


def _merge(parts: List[dict], calls: List[dict]) -> dict:
    """One reply of several calls as one card.  Finds are told per picture,
    each picture shown once with every box found in it."""
    kinds = {p["kind"] for p in parts}
    card = {"kind": parts[0]["kind"] if len(kinds) == 1 else "other",
            "text": "　|　".join(p["text"] for p in parts),
            "result": "　|　".join(p["result"] for p in parts if p["result"]),
            "ok": all(p["ok"] for p in parts),
            "thumbs": [t for p in parts for t in p.get("thumbs", [])],
            "note": "　|　".join(p["note"] for p in parts if p.get("note"))}
    if kinds == {"look", "state"}:
        card["kind"] = "look"                     # looking, and reading the hand with it
    elif kinds == {"detect"}:
        frames: Dict[str, dict] = {}
        for p, c in zip(parts, calls):
            fid = (c.get("args") or {}).get("frame_id", "")
            f = frames.setdefault(fid, {"asks": [], "thumb": None})
            f["asks"].append(p["text"].split("里找", 1)[-1])
            for t in p.get("thumbs", []):
                if f["thumb"] is None:
                    f["thumb"] = {**t, "boxes": []}
                seen = {tuple(b) for b in f["thumb"]["boxes"]}
                f["thumb"]["boxes"] += [b for b in t.get("boxes", []) if tuple(b) not in seen]
        card["text"] = f"找（{len(parts)} 次）：" + "；".join(
            f"{fid.split('.')[-1] or fid} 里找" + "、".join(f["asks"]) for fid, f in frames.items())
        card["thumbs"] = [f["thumb"] for f in frames.values() if f["thumb"]]
        card["result"] = ""
    elif kinds == {"pose"}:
        card["text"] = f"估 {len(parts)} 个位姿：" + "；".join(p["text"].split(" ", 1)[-1] for p in parts)
        card["result"] = ""
    return card


def tally(rows: List[dict]):
    """How often each tool was called: by the replies, and inside programs."""
    top: Dict[str, int] = {}
    inner: Dict[str, int] = {}
    for row in rows:
        calls = (row["args"].get("calls") or []) if row["tool"] == "batch" else [{"tool": row["tool"]}]
        for c in calls:
            top[c["tool"]] = top.get(c["tool"], 0) + 1
        for c in row.get("inner") or ():
            name = c["tool"].replace("_", ".", 1)
            inner[name] = inner.get(name, 0) + 1
    order = lambda d: sorted(d.items(), key=lambda kv: -kv[1])
    return order(top), order(inner)


def read_tool_log(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "tool" in row:
            rows.append(row)
    return rows


def load_clips(root: Path) -> Dict[str, Dict[str, dict]]:
    """{task: {profile: clip}} for every recorded clip under ``root/clips``."""
    from ruibench.tasks.instance import build

    out: Dict[str, Dict[str, dict]] = {}
    arms: Dict[str, bool] = {}
    for folder in sorted((root / "clips").glob("*")):
        meta = folder / "clip.json"
        if not meta.exists():
            continue
        clip = json.loads(meta.read_text())
        logs = [p for p in folder.glob("*.jsonl") if not p.name.endswith(".episode.jsonl")]
        if not logs:
            continue
        task = clip["task"]
        if task not in arms:
            arms[task] = len(build(task, level=clip["level"], seed=clip["seed"]).scene.arm_ids) > 1
        url = f"clips/{folder.name}"
        rows = read_tool_log(logs[0])
        noted = {}
        if (folder / "notes.json").exists():
            for n in json.loads((folder / "notes.json").read_text()):
                noted[(n["row"], n.get("call", 0))] = n["note"]
        clip["decisions"] = decisions(rows, url, arms[task], noted)
        # the episode's own count: older logs miss the reply that ended it
        clip["replies"] = max(int(clip.get("replies") or 0), len(rows))
        clip["tally"], clip["inner_tally"] = tally(rows)
        clip["video"] = f"{url}/{clip['video']}" if clip.get("video") else None
        out.setdefault(task, {})[clip["profile"]] = clip
    return out
