# Tables: build-metal/preload-start/strata-final (the fix, and HEAD 71b845ce)

### chat: 33 tokens, then 20 more (132 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=1 | STRATA_PRELOAD=1 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Start (n) | 00:00:39 (1) | 00:02:13 (2) | 00:03:40 (3) | 00:05:07 (4) | 00:06:53 (5) | 00:08:20 (6) |
| SoC before | 44.4 C | 45.0 C | 44.9 C | 45.0 C | 44.1 C | 45.0 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 1.01 GiB in 0.4 s | 1.01 GiB in 0.4 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.16 s | 3.06 s | 3.44 s | 3.41 s | 3.04 s | 5.11 s |
| Request 1: token 1 | 9.69 s | 7.84 s | 7.97 s | 7.98 s | 7.90 s | 9.45 s |
| Request 1: token 2 | 9.86 s | 8.02 s | 8.13 s | 8.14 s | 8.07 s | 9.62 s |
| Request 1: token 10 | 11.31 s | 9.43 s | 9.54 s | 9.55 s | 9.47 s | 11.02 s |
| Request 1: token 50 | 17.81 s | 15.82 s | 15.86 s | 15.86 s | 15.79 s | 17.34 s |
| Request 1: token 64 | 19.90 s | 17.90 s | 17.95 s | 18.22 s | 17.87 s | 19.41 s |
| Request 1: token 100 | 25.20 s | 23.24 s | 23.32 s | 26.42 s | 23.24 s | 24.74 s |
| Request 1: sent at | 5.17 s | 3.08 s | 3.45 s | 3.43 s | 3.05 s | 5.12 s |
| Request 1: misses per written token | 62.0 | 62.0 | 62.0 | 62.0 | 62.0 | 62.0 |
| Request 1: written | 6.38 tok/s | 6.43 tok/s | 6.45 tok/s | 5.37 tok/s | 6.45 tok/s | 6.48 tok/s |
| Request 2: token 1 | 27.63 s | 25.78 s | 25.75 s | 28.84 s | 25.69 s | 27.23 s |
| Request 2: token 2 | 27.76 s | 25.91 s | 25.89 s | 28.99 s | 25.82 s | 27.36 s |
| Request 2: token 10 | 28.98 s | 27.15 s | 27.14 s | 30.23 s | 27.06 s | 28.58 s |
| Request 2: token 50 | 35.65 s | 34.01 s | 34.00 s | 37.05 s | 33.81 s | 35.21 s |
| Request 2: token 64 | 37.74 s | 36.11 s | 36.15 s | 39.15 s | 35.91 s | 37.27 s |
| Request 2: sent at | 25.26 s | 23.30 s | 23.38 s | 26.48 s | 23.30 s | 24.80 s |
| Request 2: misses per written token | 76.3 | 76.3 | 76.4 | 76.4 | 76.4 | 76.3 |
| Request 2: written | 6.23 tok/s | 6.10 tok/s | 6.06 tok/s | 6.11 tok/s | 6.16 tok/s | 6.27 tok/s |
| Compressed while loaded | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB |
| Swap-outs | 0 | 0 | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.

### doc500: 500 tokens, then 20 more (599 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=1 | STRATA_PRELOAD=1 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Start (n) | 00:09:48 (7) | 00:11:27 (8) | 00:13:52 (9) | 00:15:22 (10) | 00:17:07 (11) | 00:18:46 (12) |
| SoC before | 44.6 C | 44.8 C | 44.2 C | 45.0 C | 44.6 C | 44.6 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 1.01 GiB in 0.4 s | 1.01 GiB in 0.4 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.11 s | 3.09 s | 3.42 s | 3.46 s | 3.13 s | 5.18 s |
| Request 1: token 1 | 15.77 s | 17.08 s | 17.27 s | 17.26 s | 15.81 s | 15.61 s |
| Request 1: token 2 | 15.91 s | 17.23 s | 17.41 s | 17.44 s | 15.96 s | 15.75 s |
| Request 1: token 10 | 17.09 s | 18.61 s | 18.70 s | 18.75 s | 17.32 s | 16.89 s |
| Request 1: token 50 | 23.27 s | 24.64 s | 24.68 s | 24.73 s | 23.34 s | 22.92 s |
| Request 1: token 64 | 25.47 s | 26.78 s | 26.84 s | 26.88 s | 25.50 s | 25.10 s |
| Request 1: token 100 | 30.48 s | 31.70 s | 31.77 s | 31.81 s | 30.47 s | 30.10 s |
| Request 1: sent at | 5.12 s | 3.11 s | 3.43 s | 3.47 s | 3.15 s | 5.19 s |
| Request 1: misses per written token | 56.0 | 57.5 | 56.9 | 56.9 | 57.5 | 55.9 |
| Request 1: written | 6.73 tok/s | 6.77 tok/s | 6.83 tok/s | 6.80 tok/s | 6.75 tok/s | 6.83 tok/s |
| Request 2: token 1 | 32.87 s | 34.08 s | 34.14 s | 34.17 s | 32.91 s | 32.58 s |
| Request 2: token 2 | 33.00 s | 34.20 s | 34.26 s | 34.29 s | 33.03 s | 32.70 s |
| Request 2: token 10 | 34.19 s | 35.37 s | 35.46 s | 35.48 s | 34.21 s | 33.88 s |
| Request 2: token 50 | 40.06 s | 41.10 s | 41.23 s | 41.28 s | 40.05 s | 39.68 s |
| Request 2: token 64 | 42.52 s | 43.52 s | 43.64 s | 43.74 s | 42.55 s | 42.14 s |
| Request 2: sent at | 30.53 s | 31.75 s | 31.83 s | 31.88 s | 30.53 s | 30.16 s |
| Request 2: misses per written token | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 |
| Request 2: written | 6.53 tok/s | 6.68 tok/s | 6.63 tok/s | 6.58 tok/s | 6.53 tok/s | 6.59 tok/s |
| Compressed while loaded | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.48 GiB | 0.00 GiB |
| Swap-outs | 0 | 0 | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.

### doc2000: 2000 tokens, then 20 more (2099 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=1 | STRATA_PRELOAD=1 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Start (n) | 00:20:24 (13) | 00:22:32 (14) | 00:24:37 (15) | 00:26:42 (16) | 00:28:52 (17) | 00:31:02 (18) |
| SoC before | 44.6 C | 44.5 C | 44.8 C | 44.8 C | 44.9 C | 44.3 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 1.01 GiB in 0.4 s | 1.01 GiB in 0.4 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.14 s | 3.06 s | 3.39 s | 3.42 s | 3.08 s | 5.12 s |
| Request 1: token 1 | 28.31 s | 26.43 s | 25.60 s | 26.65 s | 25.45 s | 27.25 s |
| Request 1: token 2 | 28.45 s | 26.57 s | 25.75 s | 26.79 s | 25.60 s | 27.40 s |
| Request 1: token 10 | 29.79 s | 27.91 s | 27.09 s | 28.13 s | 26.93 s | 28.73 s |
| Request 1: token 50 | 35.74 s | 33.85 s | 33.03 s | 34.07 s | 32.88 s | 34.69 s |
| Request 1: token 64 | 38.13 s | 36.23 s | 35.41 s | 36.45 s | 35.25 s | 37.08 s |
| Request 1: token 100 | 44.04 s | 42.23 s | 41.35 s | 42.37 s | 41.16 s | 43.01 s |
| Request 1: sent at | 5.15 s | 3.07 s | 3.40 s | 3.44 s | 3.09 s | 5.13 s |
| Request 1: misses per written token | 65.3 | 65.3 | 65.3 | 65.3 | 65.3 | 65.3 |
| Request 1: written | 6.29 tok/s | 6.27 tok/s | 6.29 tok/s | 6.30 tok/s | 6.30 tok/s | 6.28 tok/s |
| Request 2: token 1 | 46.68 s | 44.98 s | 43.96 s | 44.99 s | 43.84 s | 45.68 s |
| Request 2: token 2 | 46.80 s | 45.10 s | 44.08 s | 45.11 s | 43.96 s | 45.80 s |
| Request 2: token 10 | 47.95 s | 46.27 s | 45.23 s | 46.27 s | 45.11 s | 46.95 s |
| Request 2: token 50 | 53.92 s | 52.17 s | 51.06 s | 52.10 s | 50.94 s | 52.77 s |
| Request 2: token 64 | 55.86 s | 54.13 s | 52.99 s | 54.04 s | 52.89 s | 54.70 s |
| Request 2: sent at | 44.10 s | 42.29 s | 41.41 s | 42.42 s | 41.22 s | 43.07 s |
| Request 2: misses per written token | 57.3 | 57.3 | 57.3 | 57.3 | 57.3 | 57.3 |
| Request 2: written | 6.86 tok/s | 6.89 tok/s | 6.97 tok/s | 6.96 tok/s | 6.96 tok/s | 6.98 tok/s |
| Compressed while loaded | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB | 0.00 GiB |
| Swap-outs | 0 | 0 | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.
