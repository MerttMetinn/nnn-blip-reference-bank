# Main results (OUR MEASUREMENTS)

COCO Karpathy test, text-to-image retrieval, 5,000 images. Alpha and k selected on the tuning set (Karpathy val).

| Arm | References | R@1 | R@1 95% CI | R@5 | R@10 | seeds | (alpha, k) |
|---|---|---|---|---|---|---|---|
| A: Plain CLIP (no bank) | - | 30.49 | 29.92-31.07 | 55.96 | 66.81 | 1 | - |
| B: NNN, Flickr human captions | 29000 | 34.17 [34.05-34.28] | 33.70-34.86 | 59.40 | 69.87 | 3 | (0.625, 32), (0.625, 16), (0.5, 8) |
| C: NNN, COCO human captions (5/img) | 113369 | 38.08 [38.02-38.12] | 37.40-38.65 | 63.43 | 73.62 | 3 | (0.875, 64), (0.875, 64), (0.875, 64) |
| D: NNN, COCO human captions (1/img) | 22657 | 37.98 [37.90-38.10] | 37.29-38.52 | 63.33 | 73.47 | 3 | (0.875, 32), (0.875, 16), (0.75, 16) |
| E: NNN, COCO BLIP captions (1/img) | 22657 | 37.26 [37.21-37.31] | 36.71-37.94 | 62.89 | 73.02 | 3 | (0.75, 16), (0.75, 16), (0.75, 32) |
| F: NNN, Flickr BLIP captions (1/img) | 5800 | 34.49 [34.40-34.54] | 33.84-35.02 | 60.15 | 70.42 | 3 | (0.625, 8), (0.625, 16), (0.625, 16) |

R@1 is the mean over seeds; brackets give the min-max over seeds. The CI is a bootstrap over queries on the first seed (1,000 resamples).

Recovery ratio (E-B)/(C-B): **0.79** (seed 0: 0.81, seed 1: 0.78, seed 2: 0.78)
Recovery ratio (E-B)/(D-B): **0.81** (seed 0: 0.83, seed 1: 0.82, seed 2: 0.78)

## R@1 difference between arms (our measurements, paired bootstrap)

| Comparison | seed | Difference (points) | 95% CI |
|---|---|---|---|
| E - B | 0 | +3.03 | +2.61 to +3.45 |
| E - D | 0 | -0.61 | -0.94 to -0.29 |
| E - C | 0 | -0.71 | -1.04 to -0.39 |
| D - C | 0 | -0.10 | -0.33 to +0.14 |
| C - B | 0 | +3.74 | +3.32 to +4.18 |
| E - F | 0 | +2.91 | +2.47 to +3.32 |
| F - B | 0 | +0.12 | -0.18 to +0.45 |

## Hubness (our measurements, first seed)

| Arm | max | skewness | kurtosis | MAE |
|---|---|---|---|---|
| A | 217 | 7.37 | 102.09 | 4.86 |
| B | 85 | 3.70 | 24.74 | 3.75 |
| C | 35 | 1.34 | 4.84 | 2.51 |
| D | 25 | 1.03 | 1.88 | 2.64 |
| E | 48 | 2.31 | 11.98 | 2.91 |
| F | 96 | 4.18 | 36.73 | 3.67 |

## Values published in the paper (NOT our measurements)

Source: NNN paper, Table 1 (CLIP, COCO, image Recall@1). The paper does not state the CLIP checkpoint or the sampling unit of the bank.

| Arm | Paper R@1 | Our R@1 | Difference |
|---|---|---|---|
| A | 30.43 | 30.49 | +0.06 |
| B | 33.45 | 34.17 | +0.72 |
| C | 37.53 | 38.08 | +0.55 |

Paper hubness (CLIP, COCO; baseline -> NNN): kurtosis 59.8 -> 9.5; MAE 4.8 -> 2.6; max 162 -> 48.
