| system | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | bs8 N=1 | bs8 N=8 |
|---|---|---|---|---|---|---|---|---|
| Glance (ours) | 67 / 68 | 67 / 68 | 67 / 68 | 67 / 67 | 67 / 68 | 70 / 74 | 311 (26 q/s) | 346 (186 q/s) |
| AR-digits, incremental KV (our engine) | 446 / 472 | 462 / 509 | 517 / 546 | 550 / 602 | 569 / 644 | 603 / 680 | 828 (10 q/s) | 1405 (44 q/s) |
| AR presence-only q(t1=none), grouped blocks | 75 / 76 | 102 / 103 | 129 / 155 | 156 / 184 | 160 / 213 | 163 / 218 | 408 (19 q/s) | 545 (118 q/s) |
| AR presence-only q(t1=none), one padded pass | 67 / 68 | 67 / 68 | 67 / 68 | 67 / 68 | 68 / 68 | 71 / 75 | 314 (26 q/s) | 350 (183 q/s) |
| Qwen3.5-4B generative, vLLM + prefix cache (all requests at once) | 174 / 183 | 186 / 195 | 200 / 207 | 218 / 229 | 255 / 263 | 331 / 348 | 375 (21 q/s) | 666 (95 q/s) |
| Qwen3.5-4B generative, vLLM + prefix cache (staged: prefix first) | 174 / 183 | 197 / 206 | 213 / 220 | 235 / 243 | 276 / 283 | 349 / 362 | 375 (21 q/s) | 651 (97 q/s) |
| Qwen3.5-4B generative, vLLM, no prefix cache (all requests at once) | 171 / 183 | 196 / 204 | 236 / 245 | 309 / 316 | 461 / 474 | 755 / 774 | 373 (21 q/s) | 1419 (45 q/s) |

Cells: per-call latency median / p90 in ms (one image, N questions); bs8 columns: 8 images per call, median ms and throughput in questions/s.

| system | TTFA N=1 | TTFA N=2 | TTFA N=4 | TTFA N=8 | TTFA N=16 | TTFA N=32 | q/s N=1 | q/s N=2 | q/s N=4 | q/s N=8 | q/s N=16 | q/s N=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Glance (ours) | 67 | 67 | 67 | 67 | 67 | 70 | 14.9 | 29.9 | 59.7 | 120.1 | 238.9 | 454.7 |
| AR-digits, incremental KV (our engine) | 446 | 461 | 488 | 499 | 514 | 547 | 2.3 | 4.3 | 7.8 | 14.6 | 28.0 | 51.1 |
| AR presence-only q(t1=none), grouped blocks | 75 | 102 | 129 | 156 | 160 | 163 | 13.3 | 20.8 | 31.7 | 52.2 | 98.1 | 191.7 |
| AR presence-only q(t1=none), one padded pass | 67 | 67 | 67 | 67 | 68 | 71 | 14.9 | 29.8 | 59.5 | 118.8 | 237.2 | 448.0 |
| Qwen3.5-4B generative, vLLM + prefix cache (all requests at once) | 173 | 173 | 178 | 192 | 228 | 289 | 5.8 | 10.9 | 20.3 | 36.8 | 63.0 | 96.7 |
| Qwen3.5-4B generative, vLLM + prefix cache (staged: prefix first) | 173 | 176 | 181 | 197 | 228 | 285 | 5.8 | 10.3 | 19.0 | 34.3 | 58.5 | 92.0 |
| Qwen3.5-4B generative, vLLM, no prefix cache (all requests at once) | 171 | 182 | 215 | 277 | 401 | 644 | 5.9 | 10.3 | 17.1 | 26.1 | 35.1 | 42.3 |
