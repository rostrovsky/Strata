#!/bin/zsh
# the installed configuration's plan with the KV cache taking its memory as it fills (default) and all at once
cd /Users/rafal/Projects/strata-air/Strata
args=("${(@f)$(python3 -c "import json;print('\n'.join(json.load(open('strata-iq3_s.json'))['args']))")}")
for step in 16384 0; do
  echo "== STRATA_KV_STEP=$step"
  STRATA_KV_STEP=$step STRATA_METAL_INDEXED_QSA=1 build-metal/bin/strata $args --load-only --verbose 2>&1 | grep -E 'strata metal: (memory now|context .* on the GPU|--elastic|expert cache:|expert cache filled)|maps the cache|KV buffer size' | cut -c1-250
done
