#!/bin/zsh
# the installed configuration's plan at larger reserves: where does it stop having room for passes of 8,192 at full depth?
cd /Users/rafal/Projects/strata-air/Strata
args=("${(@f)$(python3 -c "import json;print('\n'.join(json.load(open('strata-iq3_s.json'))['args']))")}")
for r in 3 5 6 7; do
  echo "== --reserve-gib $r"
  STRATA_METAL_INDEXED_QSA=1 build-metal/bin/strata $args --reserve-gib $r --load-only 2>&1 | grep -E 'strata metal: (memory now|--elastic)' | cut -c1-260
done
