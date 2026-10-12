#!/bin/zsh
# plans.sh: the installed configuration's plan at four context sizes (--load-only: loads, fills the cache, exits)
cd /Users/rafal/Projects/strata-air/Strata
args=("${(@f)$(python3 -c "import json;print('\n'.join(json.load(open('strata-iq3_s.json'))['args']))")}")
for c in 16384 32768 65536 131072; do
  echo "== --max-context $c"
  STRATA_METAL_INDEXED_QSA=1 build-metal/bin/strata $args --max-context $c --reserve-gib 3 --load-only 2>&1 | grep -E 'memory now|context .* on the GPU|answers are written|KV buffer size|weights loaded' | cut -c1-260
done
