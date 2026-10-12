# Candidates

`desktop-128k.json` was installed on the evening of 2026-10-12 (section 8 of the report): it is the production
`strata-iq3_s.json` now. `desktop-128k-elastic.json` is not installed.

| File | Settings | Status |
| --- | --- | --- |
| `desktop-128k.json` | `--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4`, `STRATA_METAL_INDEXED_QSA=1`, the alias pi asks for | Installed. A full 130,944-token prompt ran with it without a swap-out (with the old, unaligned reads: 63.7 tok/s read, 4.54 written). The engine it points at reads aligned since later that evening |
| `desktop-128k-elastic.json` | the production file with `--prefill 8192` and `--elastic on` | Not installed. With aligned reads: one full-context run (70.1 tok/s read, 5.27 written, no swap-out) and nine shorter runs without a swap-out. Before the reads were aligned `--elastic` swapped in five runs. Its swap back-off has only run on the tiny model |

To switch: stop the server, then

```sh
cd /Users/rafal/Projects/strata-air/Strata
cp strata-iq3_s.json strata-iq3_s.json.before-elastic
cp bench/results/2026-10-12-m4-air-128k-resume/pending/desktop-128k-elastic.json strata-iq3_s.json
./run-iq3_s.sh
```

and watch it for a while with `vm_stat | grep Swapouts` before and after a session (the counter should not move).
To go back, copy `strata-iq3_s.json.before-elastic` over it. pi needs no change (its `contextWindow` is already
131072).
