| | kvh1-on | kvh2-off | kvh3-off | kvh4-on |
| --- | ---: | ---: | ---: | ---: |
| free / file cache / other programs at start (bench) | (0.56, 14.87, 13.7) | (17.77, 5.21, 6.26) | (21.86, 3.05, 4.57) | (23.09, 1.97, 4.41) |
| engine: free or file cache / held by others | ('17.8', '14.5') | ('23.1', '6.4') | ('24.4', '5.2') | ('24.5', '5.0') |
| cache for answers GiB; experts | ('10.13', '5303') | ('15.05', '7887') | ('16.28', '8530') | ('15.81', '8287') |
| cache at full depth GiB | 6.84 | 11.77 | 13.00 | 12.53 |
| long passes | 16 of 8192 | 16 of 8192 | 16 of 8192 | 16 of 8192 |
| read tok/s | 82.1 | 77.47 | 76.68 | 75.82 |
| first token s | 1594.86 | 1690.23 | 1707.7 | 1727.01 |
| wait GPU / SSD / CPU s | (1519.3, 19.0, 56.5) | (1608.0, 19.3, 63.0) | (1626.2, 19.2, 62.3) | (1649.2, 19.4, 58.4) |
| GPU busy s | 1539.1 | 2164.4 | 2225.1 | 2241.8 |
| SSD GB | 810.2 | 806.2 | 806.1 | 805.5 |
| W | 20.8 | 15.74 | 15.18 | 15.27 |
| written tok/s | 5.44 | 6.21 | 6.13 | 6.12 |
| tokens | 76 | 76 | 76 | 76 |
| per token ms GPU/SSD/CPU; misses; MB | (184.2, 120.9, 50.5, 12.7, 61.3, 287.0) | (161.5, 113.0, 37.9, 10.6, 43.6, 217.1) | (163.6, 114.8, 37.9, 11.0, 43.5, 216.7) | (164.0, 114.9, 37.8, 11.2, 43.6, 216.7) |
| token 2/10/50 after first, s | (0.254, 1.92, 9.321) | (0.256, 1.885, 8.366) | (0.255, 1.889, 8.457) | (0.256, 1.884, 8.48) |
| records | 5 | 5 | 5 | 5 |
| output | TEST_RECORD ECHO = coral-63821<|im_end|> | TEST_RECORD ECHO = coral-63821<|im_end|> | TEST_RECORD ECHO = coral-63821<|im_end|> | TEST_RECORD ECHO = coral-63821<|im_end|> |
| repeat s; same | (0.25, True) | (0.16, True) | (0.15, True) | (0.14, True) |
| compressed from launch GiB | 6.02 | 5.78 | 5.3 | 3.28 |
| free min / median while loaded | (0.01, 0.49) | (0.01, 0.45) | (0.05, 0.41) | (0.05, 0.43) |
| swap-outs (trace; bench) | (0, 0) | (0, 0) | (0, 0) | (0, 0) |
