"""The server window's request lines: one line per phase of a request - reading the prompt, thinking, answering, each
tool call - and one when it is done.

In a terminal the running phase's line is redrawn in place (a spinner, a bar, the tokens, tok/s and the time) and stays
as it ended once the next phase starts; anything else the server prints goes above it.  Piped or redirected (a log
file, a service manager), each phase prints one plain line as it ends, and a running one once a minute.
STRATA_PROGRESS=plain gives that form in a terminal too; NO_COLOR leaves the colours out."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import threading
import time

SPIN = "|/-\\"
TICK_S = 0.125              # the spinner's step; the numbers change as the request reports them
PLAIN_EVERY_S = 60.0        # piped: a running phase's line this often
BAR_W = 20
LABEL_W = 10
RATE_MIN_SPAN_S = 0.25      # a phase younger than this has its rate taken over this span

DIM, BOLD, RED, GREEN, YELLOW, CYAN, MAGENTA = "2", "1", "31", "32", "33", "36", "35"
PHASE_COLOUR = {"prompt": CYAN, "thinking": MAGENTA, "answering": GREEN, "writing": GREEN}
FINISH_COLOUR = {"stop": GREEN, "tool_calls": GREEN, "length": YELLOW}


def _vt_ok(stream) -> bool:
    """ANSI sequences work on this stream (Windows: turns the console's VT mode on)."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        k = ctypes.windll.kernel32
        h = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(mode.value & 4) or bool(k.SetConsoleMode(h, mode.value | 4))
    except Exception:
        return False


def _dur(s: float) -> str:
    if s < 10:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    m, s = divmod(int(s), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def _paint(parts, width: int, colour: bool) -> str:
    """(text, SGR) parts as one line cut to `width` visible characters."""
    out, used = [], 0
    for text, sgr in parts:
        if used >= width:
            break
        t = text[:width - used]
        used += len(t)
        out.append(f"\033[{sgr}m{t}\033[0m" if colour and sgr else t)
    return "".join(out)


class _Through:
    """sys.stdout / sys.stderr while the console draws a live line: a write first clears that line, and once a whole
    line is out the live line is drawn again below it."""

    def __init__(self, console: "Console", stream):
        self._console, self._stream = console, stream

    def write(self, s):
        return self._console.through(self._stream, s)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class Console:
    def __init__(self, live: bool = False, colour: bool = False, stream=None):
        self.live, self.colour = live, colour
        self.stream = stream                # None: whatever sys.stdout is when a line prints (tests redirect it)
        self.lock = threading.RLock()
        self.owner: Phases | None = None    # the request whose line is live (parallel requests print plain)
        self.shown = False                  # the live line is on the screen, the cursor at its end
        self.midline = False                # other output left a line unfinished: draw nothing until it ends
        self.frame = 0
        self.wake = threading.Event()

    # -- output ---------------------------------------------------------------------------------------------------
    def _out(self):
        return self.stream if self.stream is not None else sys.stdout

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((100, 24)).columns - 1)   # the last column would wrap the line

    def _clear(self):
        if self.shown:
            self.stream.write("\r\033[K")
            self.shown = False

    def _draw(self):
        if self.owner is None or self.midline:
            return
        self.stream.write("\r" + _paint(self.owner.parts(time.time(), SPIN[self.frame % len(SPIN)]),
                                        self._width(), self.colour) + "\033[K")
        self.stream.flush()
        self.shown = True

    def through(self, stream, s):
        with self.lock:
            if not s:
                return 0
            self._clear()
            if stream is not self.stream:
                self.stream.flush()
            n = stream.write(s)
            stream.flush()
            self.midline = not s.endswith("\n")
            self._draw()
            return n

    def emit(self, parts):
        """A finished line: above the live one in a terminal, plain otherwise."""
        with self.lock:
            if self.live:
                self._clear()
                self.stream.write(_paint(parts, self._width(), self.colour) + "\n")
                self.midline = False
                self._draw()
            else:
                print(_paint(parts, 10 ** 6, self.colour), file=self._out(), flush=True)

    def _tick(self):
        while True:
            self.wake.wait()
            time.sleep(TICK_S)
            with self.lock:
                if self.owner is None:
                    self.wake.clear()
                    continue
                self.frame += 1
                self._draw()

    # -- the live slot ----------------------------------------------------------------------------------------------
    def claim(self, ph: "Phases") -> bool:
        with self.lock:
            if not self.live or self.owner is not None:
                return False
            self.owner = ph
            self.wake.set()
            return True

    def finish_live(self, ph: "Phases", parts):
        """The owner's phase ended: its line stays as `parts` and the cursor moves below it."""
        with self.lock:
            if self.owner is not ph:
                return
            self._clear()
            self.stream.write(_paint(parts, self._width(), self.colour) + "\n")
            self.stream.flush()
            self.midline = False

    def release(self, ph: "Phases"):
        with self.lock:
            if self.owner is ph:
                self._clear()
                self.owner = None

    def request(self, prompt_tokens: int, max_tokens: int, budget: int | None = None) -> "Phases":
        return Phases(self, prompt_tokens, max_tokens, budget)

    def install(self):
        """Take over the terminal's output when stdout is one (see the module's doc)."""
        out = sys.stdout
        tty = bool(getattr(out, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb" and _vt_ok(out)
        self.colour = tty and not os.environ.get("NO_COLOR")
        self.live = tty and os.environ.get("STRATA_PROGRESS", "").lower() != "plain"
        if not self.live:
            return
        self.stream = out
        sys.stdout = _Through(self, out)
        sys.stderr = _Through(self, sys.stderr)
        threading.Thread(target=self._tick, name="console", daemon=True).start()
        atexit.register(self._at_exit)

    def _at_exit(self):
        with self.lock:
            self._clear()
            self.owner = None


class Phases:
    """One request's lines.  The server reports the prompt read so far, each phase as it starts and the tokens
    written; `done` ends the open phase and prints the request's last line."""

    def __init__(self, console: Console, prompt_tokens: int, max_tokens: int, budget: int | None):
        self.c = console
        self.prompt_total, self.max_tokens, self.budget = prompt_tokens, max_tokens, budget
        self.started = time.time()
        self.kind, self.label, self.tool = "prompt", "prompt", None
        self.t0, self.n0 = self.started, 0          # the open phase's start (time, tokens before it)
        self.pos, self.reused, self.n = 0, 0, 0
        self.gen_start = None                       # the first token's time
        self.last_plain = self.started
        self.live = console.claim(self)

    # -- what the server reports ------------------------------------------------------------------------------------
    def prompt(self, pos: int, total: int, reused: int = 0):
        self.pos, self.prompt_total, self.reused = pos, total or self.prompt_total, reused
        self._plain_tick()

    def tokens(self, n: int):
        self.n = n
        self._plain_tick()

    def phase(self, kind: str, n: int, tool: str | None = None):
        """A phase starts with the n-th token written: "thinking", "answering" or "tool" (`tool`: the call's name).
        The first token ends the prompt; a phase that only had an unnamed start ("writing") is renamed instead."""
        with self.c.lock:                           # the ticker draws either the old phase or the new one
            now = time.time()
            label = f"tool {tool}" if kind == "tool" else kind
            if self.kind == "prompt":
                self.pos = max(self.pos, self.prompt_total)
            elif self.kind == "writing":
                self.kind, self.label, self.tool = kind, label, tool
                return
            elif kind == self.kind and kind != "tool":
                return
            self._end(now)
            self.kind, self.label, self.tool = kind, label, tool
            self.t0, self.n0, self.n = now, max(0, n - 1), n

    def first_token(self, n: int):
        if self.kind == "prompt":
            self.phase("writing", n)

    def done(self, n: int, finish: str, cancelled: bool, hit_rate: float | None, pcie_share: float | None):
        bad = cancelled or finish in ("cancel", "disconnect", "error")
        with self.c.lock:
            now = time.time()
            self.n = n
            self._end(now, failed=bad)
            self.c.release(self)
        el = now - self.started
        gen_t = now - self.gen_start if self.gen_start is not None else 0.0
        rate = n / max(RATE_MIN_SPAN_S, gen_t) if n else 0.0
        parts = [("[strata] ", DIM), ("x " if bad else "* ", RED if bad else f"{BOLD};{GREEN}"),
                 ("done".ljust(LABEL_W) + " ", f"{BOLD};{RED if bad else GREEN}"),
                 (f"{n:,} tok in {_dur(el)}", ""), ("   ", ""), (f"{rate:.1f} tok/s", BOLD), ("   ", ""),
                 (finish, FINISH_COLOUR.get(finish, RED))]
        if cancelled and finish != "cancel":
            parts.append((" (cancelled)", RED))
        if hit_rate is not None:
            parts.append((f"   expert cache {hit_rate * 100:.1f}% hit", DIM))
            if pcie_share:
                parts.append((f" (+{pcie_share * 100:.1f}% of the routed experts over PCIe)", DIM))
        self.c.emit(parts)

    def close(self):
        """The request ended without a done line (it never started)."""
        self.c.release(self)

    # -- drawing ----------------------------------------------------------------------------------------------------
    def _end(self, now: float, failed: bool = False):
        parts = self.parts(now, "x" if failed else "+", bar=self.live)
        if self.live:
            self.c.finish_live(self, parts)
        else:
            self.c.emit(parts)
        self.last_plain = now
        if self.kind == "prompt":
            self.gen_start = now

    def _plain_tick(self):
        if not self.live and time.time() - self.last_plain >= PLAIN_EVERY_S:
            self.last_plain = time.time()
            self.c.emit(self.parts(self.last_plain, "~", bar=False))

    def parts(self, now: float, glyph: str, bar: bool = True):
        span = now - self.t0
        colour = YELLOW if self.kind == "tool" else PHASE_COLOUR.get(self.kind, GREEN)
        glyph_sgr = (f"{BOLD};{colour}" if glyph in SPIN else
                     GREEN if glyph == "+" else RED if glyph == "x" else DIM)
        if self.kind == "prompt":
            done, total = self.pos, self.prompt_total
            rate = max(0, self.pos - self.reused) / max(RATE_MIN_SPAN_S, span) if self.pos else 0.0
        else:
            done = self.n
            total = self.budget if self.kind == "thinking" and self.budget else self.max_tokens
            rate = (self.n - self.n0) / max(RATE_MIN_SPAN_S, span)
        frac = min(1.0, done / total) if total and total > 0 else 0.0
        label = self.label if len(self.label) <= LABEL_W else self.label[:LABEL_W - 1] + "~"
        p = [("[strata] ", DIM), (glyph + " ", glyph_sgr), (label.ljust(LABEL_W) + " ", colour)]
        if bar:
            fill = int(frac * BAR_W)
            head = ">" if fill < BAR_W and done else ""
            p += [("[", DIM), ("=" * fill, colour), (head, f"{BOLD};{colour}"),
                  ("-" * (BAR_W - fill - len(head)), DIM), ("] ", DIM)]
        tw = len(f"{total:,}") if total and total > 0 else 1
        unit = " tok budget" if self.kind == "thinking" and self.budget else " tok"     # #984: not max_tokens
        p += [(f"{int(frac * 100):3d}%   ", ""),
              (f"{done:>{tw},} of {total:,}{unit}" if total and total > 0 else f"{done:,} tok", ""), ("   ", ""),
              (f"{rate:6.1f} tok/s", BOLD), ("   ", ""), (_dur(span), DIM)]
        if self.kind == "prompt" and self.reused:
            p.append((f"   ({self.reused:,} reused)", DIM))
        return p


CONSOLE = Console()
