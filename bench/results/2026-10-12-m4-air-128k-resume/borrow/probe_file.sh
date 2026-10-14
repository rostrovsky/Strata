#!/bin/zsh
# probe_file.sh: what macOS gives when the file cache is above the floor it keeps: no engine; 3 GiB of the model's
# second shard read through the file cache, then givable_probe until macOS compresses (up to 14 GiB), then those
# pages dropped from the file cache again (resident --evict)
cd /Users/rafal/Projects/strata-air/Strata
F=../Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
for k in 1 2; do
    echo "== $(date +%H:%M:%S) round $k: reading 3 GiB of the shard through the file cache"
    dd if=$F of=/dev/null bs=16m count=192 2>&1 | tail -1
    sleep 5
    build-metal/borrow/bin/givable_probe 64 50 16 14336
    sleep 2
    build-metal/resume-128k/resident --evict -q $F
    echo "dropped; $(build-metal/resume-128k/resident $F | tail -1)"
    sleep 20
done
