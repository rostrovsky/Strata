# Candidates

Both files are the production `strata-iq3_s.json` of 2026-10-12 with the 128k settings added.

**`desktop-128k.json` was installed on the evening of 2026-10-12**, after a full 130,944-token prompt ran with it
without a swap-out; pi's `qwen-local` provider has a 131,072-token context since then. See section 8 of
[the report](../README.md) for the run, the checks and how to go back. `desktop-128k-elastic.json` is not in use.
The table below is the afternoon's.

| File | Adds | Status |
| --- | --- | --- |
| `desktop-128k.json` | `--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4`, `STRATA_METAL_INDEXED_QSA=1`, the alias pi asks for | Prompts up to 16,200 tokens ran with this build and these settings without a swap-out. A full 131,072-token prompt was last read by the morning's build, whose plan was 0.89 GiB less careful, and swapped 61.7 MiB. Not repeated with the new plan |
| `desktop-128k-elastic.json` | the same with `--elastic on` and `--reserve-gib 6` | Experimental. `--elastic` swapped at `--reserve-gib 4` (see the report). 6 is a guess: it was not run |

To try one: stop the server, copy the file over `strata-iq3_s.json` (keep the old one), start `run-iq3_s.sh`, and
watch `vm_stat | grep Swapouts` before and after. For pi, `contextWindow` of the `qwen-local` model in
`~/.pi/agent/models.json` goes to 131072; `pi-smoke.txt` and the smoke command are in
[OPTIMIZATION.md](../../../../OPTIMIZATION.md).
