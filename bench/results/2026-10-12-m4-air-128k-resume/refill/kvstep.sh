#!/bin/zsh
# The two ways the expert cache gives memory to the KV cache while it is there (as kv-cache/det5.sh: a step of 1,024
# cells, so that a short conversation crosses one), with the cache read back whole afterwards (--elastic-refill all,
# as before) and left to fill as its experts are asked for (the default now).  The installed configuration, no cache
# cap: with a cap the KV cache's step is paid from what the cap left unused and the cache is not touched (a first
# run with --expert-cache-cap 7000 showed exactly that: "6.84 -> 6.84 GiB, 0.00 s").
# determinism.py compares IDs and logits of the second run of each pair with the first.
cd /Users/rafal/Projects/strata-air/Strata
O=build-metal/refill/kvstep2
D=(.venv/bin/python -u metal/tools/determinism.py --config strata-iq3_s.json --once)
A="read-back:STRATA_KV_STEP=1024 --elastic-refill all --verbose"
B="as-asked-for:STRATA_KV_STEP=1024 --verbose"
echo "== 900 tokens in one short pass, 64 written; the cache gives memory before the pass"
$D --out $O/kvstep-900 --tokens 900 --new 64 $A $B
echo "== 700 tokens, 120 written; the cache gives memory in the middle of the answer"
$D --out $O/kvstep-700 --tokens 700 --new 120 $A $B
for d in kvstep-900 kvstep-700; do
    for n in read-back as-asked-for; do
        echo "$d $n: $(grep -a 'the conversation reaches' $O/$d/$n-engine.txt | sed 's/strata metal: //')"
        .venv/bin/python - $O/$d/$n.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["requests"][0]
at = r["token_s"]
gaps = [(round(b - a, 2), i + 2) for i, (a, b) in enumerate(zip(at, at[1:]))]
print("    first token %.2f s, token 50 %.2f s, last (%d) %.2f s; longest wait between two tokens %.2f s (before token %d)"
      % (at[0], at[49], len(at), at[-1], *max(gaps)))
PY
    done
done
