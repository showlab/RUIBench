"""The words the report pages show people (in Chinese): interfaces, kinds of task,
steps, sizes, the rules of each task."""

from __future__ import annotations

PROFILES = ("rui.step/v1", "rui.pose/v1", "rui.point/v1", "rui.code/v1")


#: how each base interface is introduced: its name, what it copies, and what
#: the model says each time
INTERFACES = {
    "rui.step/v1": ("语义步", "仿 Show-Harness",
                    "每次回复只说一个离散动作：前后左右上下、转腕、倾斜、夹或松，步长分微、细、粗三档；"
                    "同一步可连走 1–5 次。每走一步就把控制交回模型，并回传两台相机的画面。",
                    "一次回复看（同时读自己的位置），一次找，一次估位姿，得到目标；再按差值一步一步走过去"),
    "rui.pose/v1": ("位姿", "仿 Robocurve / Inspect Robots",
                    "每次说夹爪要到达的绝对位置和朝向（站台坐标，米、度），动作做完再交回。",
                    "一次回复看，一次找，一次估位姿；然后 move_to（估计位置 + 偏移）"),
    "rui.point/v1": ("指点", "仿 MOKA / Gemini-ER",
                     "在某台相机刚拍的图上点一个像素，再给出比那一点的表面高多少（或穿过的高度）和朝向。",
                     "一次回复看，一次找，一次估位姿；目标就在物体正上方时再一次 vision_lift 量表面高度（量到的点离瞄准处超过 3 mm 就改按高度指）；然后在图上点那个位置"),
    "rui.code/v1": ("程序", "仿 CaP-X",
                    "写一段 Python，里面调用的工具和位姿接口完全相同；整段程序跑完才交回。",
                    "程序一：看、找、估位姿，存进变量 seen（图里要找的东西没问到时再写一段）；程序二：move_to(at(seen[…], 偏移))"),
}


#: a card's tag: what kind of call it is
KIND_TAGS = {"look": "看", "detect": "找", "pose": "估", "lift": "量", "state": "读", "step": "步",
             "move": "动", "point": "指", "grip": "夹", "wait": "等", "code": "写", "done": "完",
             "stop": "停", "other": "·"}


#: a step's words, as the model says them
STEP_WORDS = {"forward": "前", "back": "后", "left": "左", "right": "右", "up": "上", "down": "下",
              "rotate_ccw": "逆时针转", "rotate_cw": "顺时针转", "tilt_forward": "前倾",
              "tilt_back": "后仰", "tilt_left": "左倾", "tilt_right": "右倾",
              "grasp": "夹紧", "release": "松开"}


SIZE_WORDS = {"micro": "微", "fine": "细", "coarse": "粗"}


ENGINE_NAMES = {"IsaacEngine": "Isaac Sim", "MockEngine": "mock"}


#: the tasks' rules, in words (a broken rule fails a cell that otherwise succeeded)
RULES = {"on_the_road_on_red": "红灯时还在马路上", "only_when_simon_says": "没说“Simon says”也照做了",
         "wait_for_it_to_stop": "没等它停下就动手", "nothing_knocked_over": "碰倒了东西",
         "nothing_eaten": "有东西被吃掉了", "leave_what_is_on_it": "动了上面放着的东西",
         "no_strip_in_a_beaker": "试纸掉进了烧杯", "one_shell_only": "翻了不止一个壳",
         "only_one_cover_lifted": "掀了不止一个盖子", "weighed_within_allowance": "称的次数超了",
         "fair_play": "没守游戏规则"}
