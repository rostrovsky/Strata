# Configurations

Both were installed, one after the other; the second is the production `strata-iq3_s.json` now.

| File | Settings | Status |
| --- | --- | --- |
| `desktop-128k.json` | `--max-context 131072 --kv f16 --prefill 4096 --reserve-gib 4`, `STRATA_METAL_INDEXED_QSA=1`, the alias pi asks for | Installed on the evening of 2026-10-12 (section 8 of the report), replaced that night. The same file is `../installed/strata-iq3_s.before-elastic.json` |
| `desktop-128k-elastic.json` | the same with `--prefill 8192 --reserve-gib 3 --elastic on` | **Installed** at the owner's request on the night of 2026-10-12 (section 10): first with `--reserve-gib 4`, then, after the engine stopped leaving the model in the OS file cache, with 3. Measured without a swap-out: a full-context prompt (at 4, section 9), a 24,000-token prompt and a three-request session through the server (at 3) |

To go back to the plain configuration: stop the server, then

```sh
cd /Users/rafal/Projects/strata-air/Strata
cp bench/results/2026-10-12-m4-air-128k-resume/installed/strata-iq3_s.before-elastic.json strata-iq3_s.json
./run-iq3_s.sh
```

pi needs no change (its `contextWindow` is 131072 for both).

2026-10-13: the alias `qwen3.8-27b-iq3_s` is gone from the production config and from `desktop-128k-elastic.json`.
It was only there because pi's `qwen-local` entry asked for that name, and the name was wrong: Qwen3.8-27B is
another, dense model; this is Qwen3.8-Flash-Next (125B + 51B n-gram embeddings). pi now asks for the config's
`model_name`, `qwen3.8-flash-next-iq3_s` (`~/.pi/agent/models.json`, backup beside it). The server answers
requests under any model name, so nothing that still uses the old one stops working; the older files in
`installed/` keep the alias as they were.
