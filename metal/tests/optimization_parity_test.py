"""Compare Metal optimization builds on a tiny qwen4exp model, including logits and speculative verification.

Save the reference executable before rebuilding, then pass --reference, --engine and --model.
The generated model used by serve_protocol_test.py is sufficient; no real-model weights are required.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bench import Engine


def run(exe, model, env, log, speculative=False):
    args = ["--native", model, "--max-context", "1024", "--prefill", "256", "--kv", "f16"]
    if speculative:
        args += ["--lookup", "1", "--spec", "3"]
    engine = Engine(exe, args, log, env)
    try:
        if speculative:
            # Force a sequence present in the prompt so lookup exercises a four-output verification pass.
            ids = [1, 2, 3, 4] * 40
            force = ".".join(map(str, [1, 2, 3, 4] * 3))
            result = engine.gen(ids, 12, " probe=1 force=" + force)
            assert result["drafts_offered"] > 0, "speculative path was not exercised"
            return [result]
        return [engine.gen([1 + (i * 7 + n) % 120 for i in range(n)], 12, " probe=1")
                for n in (15, 64, 129, 257, 513)]
    except BaseException:
        print(log.read_text()[-8000:], file=sys.stderr, flush=True)
        raise
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    variants = {
        "default": {"STRATA_METAL_INDEXED_QSA": "0", "STRATA_METAL_QSA_SIMD_SORT": "0"},
        "indexed": {"STRATA_METAL_INDEXED_QSA": "1", "STRATA_METAL_QSA_SIMD_SORT": "0"},
        "simd": {"STRATA_METAL_INDEXED_QSA": "1", "STRATA_METAL_QSA_SIMD_SORT": "1"},
    }
    with tempfile.TemporaryDirectory(prefix="strata-parity-") as temp:
        for speculative in (False, True):
            reference = run(args.reference, args.model, variants["default"],
                            Path(temp) / "reference.log", speculative)
            for label, env in variants.items():
                candidate = run(args.engine, args.model, env, Path(temp) / (label + ".log"), speculative)
                for index, (want, got) in enumerate(zip(reference, candidate)):
                    assert got["ids"] == want["ids"], (label, speculative, index, "token mismatch")
                    # Probe rows contain the fed-token logit and the three largest logits, printed to six decimals.
                    assert got["probes"] == want["probes"], (label, speculative, index, "logit mismatch")
                print(f"{label}, speculative={speculative}: {len(candidate)} cases, identical IDs and probes", flush=True)


if __name__ == "__main__":
    main()
