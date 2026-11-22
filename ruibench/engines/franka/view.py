"""The operator's window on the real station: both cameras, live, and the key that stops the arm.

A pygame window, as Show-Harness's ``core/ui/live_view.py`` (OpenCV here is a
headless build).  One thread draws it and reads its keys, and goes on while a
model thinks and while the arm moves.  The pictures are the cameras' newest,
through the robot model's lens, as the tools get them.

**P** stops the arm where it is and the episode's clock with it, and **P** again
goes on with what was under way (``FrankaEngine.pause``).  **F**, in an episode the
runner watches, fails it: the arm stops for good, and the runner goes on to the next
(``FrankaEngine.fail_by_operator``).  A window hears the keyboard only while it has the
focus, so it says when it has not.  It stays open
as long as the arm may move: closing it would leave no way out of a pause.

It is a convenience, not a safety device: it needs this machine, the window's
focus and the NUC's server to work.  The robot's own stop is the one that always
does.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

#: reads the keys this often, so that P is acted on at once; draws less often
KEYS_HZ = 50.0
DRAW_HZ = 12.0
#: both pictures this high, the fixed camera's keeping its 4:3
PICTURE_HEIGHT = 420
GAP = 10
STRIP = 44
DOING_STRIP = 24
FOCUS_STRIP = 24
#: set to 0 for no window, to 1 for one on a stand-in robot too
VIEW_ENV = "RUIBENCH_FRANKA_VIEW"

BACKGROUND = (28, 30, 34)
INK = (235, 235, 235)
RUNNING = (39, 105, 62)
PAUSED = (214, 134, 16)
FAULT = (190, 42, 42)


def can_show() -> Tuple[bool, str]:
    """Can this machine open the window?  Said before anything moves."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame

        pygame.display.init()
        driver = pygame.display.get_driver()
        pygame.display.quit()
    except Exception as exc:            # no pygame, no display: said the same way
        return False, (f"cannot open the operator's window, and P will not pause the arm "
                       f"({type(exc).__name__}: {exc}); {VIEW_ENV}=0 runs without it")
    return True, f"the operator's window opens on {driver}: P pauses the arm"


class LiveView:
    """The window (the module docstring).  ``pictures()`` gives the newest pictures
    by name, ``state()`` the words for its top strip and whether the arm is paused
    or has faulted, ``toggle()`` is what P does, ``doing()`` what the agent has under way
    (``steps.py``), ``fail()`` what F does, if it does anything here; any of them may fail,
    and the window goes on."""

    def __init__(self, title: str, pictures: Callable[[], Dict[str, Any]],
                 state: Callable[[], Tuple[str, str]], toggle: Callable[[], None],
                 doing: Callable[[], str] = str,
                 fail: Optional[Callable[[], None]] = None) -> None:
        self.title, self._pictures, self._state, self._toggle = title, pictures, state, toggle
        self._doing, self._fail = doing, fail
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._shown: Dict[str, Any] = {}
        self.problem = ""

    def start(self) -> "LiveView":
        self._thread = threading.Thread(target=self._run, name="franka-view", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- the window's thread: every pygame call is here -------------------------------------
    def _run(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame

            pygame.display.init()
            pygame.font.init()
            font = pygame.font.Font(None, 30)
            small = pygame.font.Font(None, 22)
            width = 2 * GAP + PICTURE_HEIGHT * 4 // 3 + GAP + PICTURE_HEIGHT
            height = STRIP + DOING_STRIP + FOCUS_STRIP + PICTURE_HEIGHT + GAP
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(self.title)
            _keep_on_top(pygame)
            drawn = 0.0
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self._act(self._toggle)
                        drawn = 0.0                     # and show it now
                    elif (event.type == pygame.KEYDOWN and event.key == pygame.K_f
                          and self._fail is not None):
                        self._act(self._fail)
                        drawn = 0.0
                if time.monotonic() - drawn >= 1.0 / DRAW_HZ:
                    drawn = time.monotonic()
                    self._draw(pygame, screen, font, small)
                time.sleep(1.0 / KEYS_HZ)
            pygame.display.quit()
        except Exception as exc:                        # the arm goes on, and says so at the check
            self.problem = f"{type(exc).__name__}: {exc}"

    def _act(self, call):
        try:
            return call()
        except Exception as exc:
            self.problem = f"{type(exc).__name__}: {exc}"
            return None

    def _draw(self, pygame, screen, font, small) -> None:
        shown = self._act(self._pictures)
        if shown:
            self._shown = shown
        words, status = self._act(self._state) or ("no word from the arm", "fault")
        colour = {"paused": PAUSED, "fault": FAULT}.get(status, RUNNING)
        width = screen.get_width()
        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, colour, (0, 0, width, STRIP))
        screen.blit(font.render(words, True, INK), (GAP, 11))
        screen.blit(small.render(self._act(self._doing) or "", True, INK), (GAP, STRIP + 4))
        if pygame.key.get_focused():
            line, back = "P pauses and goes on", BACKGROUND
            if self._fail is not None:
                line += "; F fails this interface and goes on to the next"
        else:
            line, back = "KEYS INACTIVE: click this window, or P does nothing", FAULT
        pygame.draw.rect(screen, back, (0, STRIP + DOING_STRIP, width, FOCUS_STRIP))
        screen.blit(small.render(line, True, INK), (GAP, STRIP + DOING_STRIP + 4))
        x, y = GAP, STRIP + DOING_STRIP + FOCUS_STRIP
        for name in sorted(self._shown):                # agentview, then wrist
            image = self._shown[name]
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            h, w = image.shape[:2]
            size = (round(w * PICTURE_HEIGHT / h), PICTURE_HEIGHT)
            screen.blit(pygame.transform.smoothscale(surface, size), (x, y))
            if status == "paused":
                pygame.draw.rect(screen, PAUSED, (x, y, size[0], size[1]), 4)
            x += size[0] + GAP
        pygame.display.flip()


#: pygame's wrapper owns the SDL window: collected, it destroys the window
#: (Show-Harness's ``_WINDOW_REFS``, verified on pygame 2.6.1)
_WINDOWS: list = []


def _keep_on_top(pygame) -> bool:
    """Ask for the window to stay above the others, best effort.  pygame 2.6 has no
    call for it, its SDL does: in the library pygame has loaded (``/proc/self/maps``),
    which a second copy of SDL could not reach the window from."""
    try:
        import ctypes

        from pygame._sdl2.video import Window

        window = Window.from_display_module()
        _WINDOWS.append(window)
        with open("/proc/self/maps", encoding="utf-8") as maps:
            libraries = {line.rsplit(maxsplit=1)[-1].strip() for line in maps
                         if "libSDL2-" in line or "libSDL2." in line}
        for library in libraries:
            sdl = ctypes.CDLL(library)
            sdl.SDL_GetWindowFromID.argtypes = [ctypes.c_uint32]
            sdl.SDL_GetWindowFromID.restype = ctypes.c_void_p
            sdl.SDL_SetWindowAlwaysOnTop.argtypes = [ctypes.c_void_p, ctypes.c_int]
            sdl.SDL_SetWindowAlwaysOnTop.restype = None
            found = sdl.SDL_GetWindowFromID(int(window.id))
            if found:
                sdl.SDL_SetWindowAlwaysOnTop(found, 1)
                return True
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        pass
    return False


__all__ = ["LiveView", "VIEW_ENV", "can_show"]
