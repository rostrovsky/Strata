"""serve/console.py: one line per phase of a request, redrawn in place in a terminal, plain when piped."""

from __future__ import annotations

import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.console import Console, _Through  # noqa: E402

ANSI = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def screen(raw: str) -> list[str]:
    """What a terminal shows for `raw`: a carriage return starts the line over, ESC[K clears the rest of it."""
    rows, cur = [], ""
    for chunk in re.split(r"(\r|\n)", raw.replace("\033[K", "")):
        if chunk == "\r":
            cur = ""
        elif chunk == "\n":
            rows.append(cur)
            cur = ""
        else:
            cur += ANSI.sub("", chunk)
    return rows + ([cur] if cur else [])


def tool_request(c: Console):
    ph = c.request(96653, 29268)
    ph.prompt(96440, 96653, 96440)
    ph.first_token(1)
    ph.phase("thinking", 1)
    ph.tokens(304)
    ph.phase("tool", 309, "read")
    ph.tokens(328)
    ph.done(328, "stop", False, 0.887, None)


class Plain(unittest.TestCase):
    def test_one_line_per_phase(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tool_request(Console())
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 4, lines)
        self.assertRegex(lines[0], r"^\[strata\] \+ prompt +100% +96,653 of 96,653 tok .* tok/s .*\(96,440 reused\)$")
        self.assertRegex(lines[1], r"^\[strata\] \+ thinking +1% +304 of 29,268 tok +[\d.]+ tok/s")
        self.assertRegex(lines[2], r"^\[strata\] \+ tool read +1% +328 of 29,268 tok")
        self.assertRegex(lines[3], r"^\[strata\] \* done +328 tok in .* tok/s +stop +expert cache 88\.7% hit$")
        self.assertNotIn("\033", out.getvalue())

    def test_unnamed_start_is_renamed_not_a_new_line(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ph = Console().request(10, 100)
            ph.first_token(1)              # the first token gave no event yet
            ph.phase("answering", 2)
            ph.done(5, "stop", False, None, None)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("answering", lines[1])

    def test_cancelled_while_reading(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ph = Console().request(1000, 100)
            ph.prompt(400, 1000)
            ph.done(0, "cancel", True, None, None)
        lines = out.getvalue().splitlines()
        self.assertRegex(lines[0], r"^\[strata\] x prompt +40% +400 of 1,000 tok")
        self.assertRegex(lines[1], r"^\[strata\] x done +0 tok .* cancel")


class Live(unittest.TestCase):
    def test_redrawn_in_place_and_other_output_goes_above(self):
        term = io.StringIO()
        c = Console(live=True, colour=True, stream=term)
        ph = c.request(96653, 29268)
        ph.prompt(50000, 96653)
        c._draw()
        through = _Through(c, term)
        print("[strata] some other message", file=through)
        ph.first_token(1)
        ph.phase("thinking", 1)
        c._draw()
        ph.tokens(40)
        c._draw()
        ph.done(40, "stop", False, None, None)
        rows = screen(term.getvalue())
        self.assertEqual(rows[0], "[strata] some other message")
        self.assertRegex(rows[1], r"^\[strata\] \+ prompt +\[=+\] 100%")
        self.assertRegex(rows[2], r"^\[strata\] \+ thinking +\[>-+\] +0% +40 of 29,268 tok")
        self.assertRegex(rows[3], r"^\[strata\] \* done +40 tok")
        self.assertEqual(len(rows), 4, rows)
        self.assertIn("\033[", term.getvalue())
        self.assertIsNone(c.owner)

    def test_a_second_request_at_once_prints_plain(self):
        term = io.StringIO()
        c = Console(live=True, stream=term)
        first, second = c.request(10, 100), c.request(10, 100)
        self.assertTrue(first.live)
        self.assertFalse(second.live)
        second.done(0, "stop", False, None, None)
        first.done(0, "stop", False, None, None)
        self.assertTrue(c.request(10, 100).live)

    def test_lines_fit_the_width(self):
        term = io.StringIO()
        c = Console(live=True, stream=term)
        c._width = lambda: 40
        ph = c.request(96653, 29268)
        ph.prompt(96440, 96653, 96440)
        c._draw()
        self.assertTrue(all(len(r) <= 40 for r in screen(term.getvalue())))


if __name__ == "__main__":
    unittest.main()
