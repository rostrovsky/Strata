# Tables: build-metal/preload-start/strata (before the fix of the staged read-ahead)

### chat: 33 tokens, then 20 more (132 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: |
| Start (n) | 19:27:17 (1) | 19:28:57 (2) | 19:30:26 (3) | 19:31:54 (4) |
| SoC before | 41.7 C | 44.8 C | 44.7 C | 44.4 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.12 s | 3.15 s | 3.15 s | 5.19 s |
| Request 1: token 1 | 9.93 s | 8.29 s | 8.21 s | 9.80 s |
| Request 1: token 2 | 10.09 s | 8.45 s | 8.37 s | 9.96 s |
| Request 1: token 10 | 11.51 s | 9.87 s | 9.84 s | 11.39 s |
| Request 1: token 50 | 18.03 s | 16.42 s | 16.40 s | 17.88 s |
| Request 1: token 64 | 20.15 s | 18.58 s | 18.52 s | 20.03 s |
| Request 1: token 100 | 25.60 s | 24.16 s | 24.04 s | 25.50 s |
| Request 1: sent at | 5.13 s | 3.16 s | 3.17 s | 5.20 s |
| Request 1: misses per written token | 62.0 | 62.0 | 62.0 | 62.0 |
| Request 1: written | 6.32 tok/s | 6.24 tok/s | 6.25 tok/s | 6.30 tok/s |
| Request 2: token 1 | 28.10 s | 27.08 s | 26.60 s | 28.05 s |
| Request 2: token 2 | 28.23 s | 27.22 s | 26.73 s | 28.19 s |
| Request 2: token 10 | 29.47 s | 28.48 s | 27.99 s | 29.47 s |
| Request 2: token 50 | 36.37 s | 35.69 s | 34.86 s | 36.39 s |
| Request 2: token 64 | 38.49 s | 37.95 s | 37.00 s | 38.52 s |
| Request 2: sent at | 25.66 s | 24.23 s | 24.10 s | 25.57 s |
| Request 2: misses per written token | 76.4 | 76.4 | 76.3 | 76.3 |
| Request 2: written | 6.06 tok/s | 5.80 tok/s | 6.06 tok/s | 6.02 tok/s |
| Compressed while loaded | 0.25 GiB | 0.05 GiB | 0.00 GiB | 0.00 GiB |
| Swap-outs | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.

### doc2000: 2000 tokens, then 20 more (2099 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: |
| Start (n) | 19:33:39 (5) | 19:35:52 (6) | 19:38:38 (7) | 19:41:20 (8) |
| SoC before | 44.5 C | 44.8 C | 44.5 C | 44.5 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.17 s | 3.11 s | 3.07 s | 5.18 s |
| Request 1: token 1 | 27.50 s | 35.25 s | 35.74 s | 27.56 s |
| Request 1: token 2 | 27.65 s | 35.42 s | 35.89 s | 27.71 s |
| Request 1: token 10 | 28.99 s | 36.77 s | 37.26 s | 29.05 s |
| Request 1: token 50 | 35.18 s | 42.94 s | 43.52 s | 35.06 s |
| Request 1: token 64 | 37.68 s | 45.43 s | 46.05 s | 37.46 s |
| Request 1: token 100 | 43.86 s | 51.62 s | 52.38 s | 43.44 s |
| Request 1: sent at | 5.19 s | 3.12 s | 3.08 s | 5.20 s |
| Request 1: misses per written token | 68.9 | 68.6 | 66.6 | 65.3 |
| Request 1: written | 6.05 tok/s | 6.05 tok/s | 5.95 tok/s | 6.24 tok/s |
| Request 2: token 1 | 46.72 s | 54.57 s | 55.30 s | 46.21 s |
| Request 2: token 2 | 46.85 s | 54.70 s | 55.44 s | 46.33 s |
| Request 2: token 10 | 48.06 s | 55.91 s | 56.75 s | 47.50 s |
| Request 2: token 50 | 54.14 s | 62.06 s | 62.82 s | 53.41 s |
| Request 2: token 64 | 56.18 s | 64.09 s | 64.83 s | 55.37 s |
| Request 2: sent at | 43.93 s | 51.68 s | 52.44 s | 43.50 s |
| Request 2: misses per written token | 60.7 | 60.3 | 58.5 | 57.3 |
| Request 2: written | 6.66 tok/s | 6.62 tok/s | 6.61 tok/s | 6.87 tok/s |
| Compressed while loaded | 0.00 GiB | 0.00 GiB | 0.63 GiB | 0.00 GiB |
| Swap-outs | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.

### doc500: 500 tokens, then 20 more (599 reused)

|  | STRATA_PRELOAD=all | STRATA_PRELOAD=0 | STRATA_PRELOAD=1 | STRATA_PRELOAD=2 | STRATA_PRELOAD=2 | STRATA_PRELOAD=1 | STRATA_PRELOAD=0 | STRATA_PRELOAD=all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start (n) | 19:43:44 (1) | 19:45:33 (2) | 19:47:25 (3) | 19:49:09 (4) | 19:50:53 (5) | 19:52:36 (6) | 19:54:15 (7) | 19:56:11 (8) |
| SoC before | 45.0 C | 44.6 C | 44.8 C | 44.8 C | 44.6 C | 44.8 C | 45.0 C | 44.7 C |
| Expert cache | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB | 6493 MiB |
| Filled at the start | 6.33 GiB in 2.1 s | 0.00 GiB in 0.0 s | 1.01 GiB in 0.4 s | 2.02 GiB in 0.7 s | 2.02 GiB in 0.7 s | 1.01 GiB in 0.4 s | 0.00 GiB in 0.0 s | 6.33 GiB in 2.1 s |
| READY | 5.06 s | 2.99 s | 3.38 s | 3.75 s | 3.71 s | 3.43 s | 3.02 s | 5.08 s |
| Request 1: token 1 | 15.96 s | 23.00 s | 15.80 s | 15.79 s | 15.64 s | 15.79 s | 23.07 s | 15.75 s |
| Request 1: token 2 | 16.10 s | 23.14 s | 15.95 s | 15.95 s | 15.78 s | 15.94 s | 23.22 s | 15.89 s |
| Request 1: token 10 | 17.26 s | 24.54 s | 17.27 s | 17.20 s | 17.02 s | 17.23 s | 24.61 s | 17.04 s |
| Request 1: token 50 | 23.33 s | 30.68 s | 23.37 s | 23.33 s | 23.06 s | 23.22 s | 30.75 s | 23.04 s |
| Request 1: token 64 | 25.54 s | 32.89 s | 25.63 s | 25.52 s | 25.26 s | 25.38 s | 32.95 s | 25.21 s |
| Request 1: token 100 | 30.58 s | 37.92 s | 30.76 s | 30.55 s | 30.29 s | 30.36 s | 37.98 s | 30.18 s |
| Request 1: sent at | 5.07 s | 3.00 s | 3.39 s | 3.76 s | 3.73 s | 3.44 s | 3.03 s | 5.09 s |
| Request 1: misses per written token | 55.9 | 57.5 | 56.9 | 56.5 | 56.5 | 56.9 | 57.5 | 56.0 |
| Request 1: written | 6.77 tok/s | 6.63 tok/s | 6.62 tok/s | 6.71 tok/s | 6.76 tok/s | 6.80 tok/s | 6.64 tok/s | 6.86 tok/s |
| Request 2: token 1 | 32.98 s | 40.34 s | 33.40 s | 32.93 s | 32.73 s | 32.76 s | 40.39 s | 32.56 s |
| Request 2: token 2 | 33.10 s | 40.47 s | 33.53 s | 33.06 s | 32.85 s | 32.88 s | 40.51 s | 32.68 s |
| Request 2: token 10 | 34.29 s | 41.65 s | 34.72 s | 34.24 s | 34.04 s | 34.05 s | 41.70 s | 33.85 s |
| Request 2: token 50 | 40.17 s | 47.52 s | 40.60 s | 40.09 s | 39.95 s | 39.85 s | 47.57 s | 39.70 s |
| Request 2: token 64 | 42.63 s | 50.00 s | 43.06 s | 42.56 s | 42.41 s | 42.29 s | 50.04 s | 42.17 s |
| Request 2: sent at | 30.64 s | 37.98 s | 30.83 s | 30.61 s | 30.35 s | 30.42 s | 38.04 s | 30.24 s |
| Request 2: misses per written token | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 |
| Request 2: written | 6.53 tok/s | 6.53 tok/s | 6.52 tok/s | 6.54 tok/s | 6.51 tok/s | 6.61 tok/s | 6.53 tok/s | 6.55 tok/s |
| Compressed while loaded | 0.60 GiB | 0.00 GiB | 0.00 GiB | 0.04 GiB | 0.08 GiB | 0.00 GiB | 0.00 GiB | 0.22 GiB |
| Swap-outs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Same ids in every start: request 1 True, request 2 True.
