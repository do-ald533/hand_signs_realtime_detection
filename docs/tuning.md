# Hyperparameter selection

Everything here was measured on the **validation** split (177 clips). The test split was not read
until the final configuration was fixed. Each GRU row is the mean of seeds 0 and 1; with 177 clips a
single accuracy is only known to about ±3 points, so differences of a few points are noise.

## Baseline (mean landmarks + logistic regression)

| C | Val top-1 | Val top-5 |
| ---: | ---: | ---: |
| 0.0005 | 4.0% | 14.7% |
| 0.005 | 9.6% | 29.4% |
| 0.05 | 18.6% | 41.2% |
| 0.5 | 23.2% | 51.4% |
| 1 (first try) | 24.9% | 52.0% |
| 5 | 28.2% | 49.2% |
| 20 | 28.8% | 51.4% |
| **100** | **29.9%** | **52.0%** |
| 1000 | 22.6% | 46.9% |

## GRU

All rows: two layers, AdamW, cosine schedule, label smoothing 0.1, rotation ±10°, scale 0.9–1.1,
mirror with left/right swap. "Aug" is time-crop / per-frame hand dropout / landmark noise.

| Config | Hidden | Dropout | Weight decay | Pooling | Aug (crop / hand drop / noise) | Epochs | Val top-1 | Val top-5 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| A: first try | 256 | 0.3 | 0.01 | last | none | 100 | 19.8% | 37.3% |
| B: smaller, more regularized | 128 | 0.5 | 0.05 | last | none | 100 | 20.3% | 40.4% |
| C: B + augmentation | 128 | 0.5 | 0.05 | last | 0.2 / 0.1 / 0.02 | 100 | 23.2% | 47.5% |
| D: C with 64 units | 64 | 0.5 | 0.05 | last | 0.2 / 0.1 / 0.02 | 100 | 19.5% | 46.0% |
| E: C with lr 1e-3 | 128 | 0.5 | 0.05 | last | 0.2 / 0.1 / 0.02 | 100 | 20.9% | 49.2% |
| F: C, stronger aug | 128 | 0.5 | 0.05 | last | 0.3 / 0.25 / 0.05 | 100 | 24.3% | 48.3% |
| G: F, mean pooling | 128 | 0.5 | 0.05 | mean | 0.3 / 0.25 / 0.05 | 100 | 29.1% | 57.6% |
| H: F, stronger aug | 128 | 0.5 | 0.05 | last | 0.4 / 0.35 / 0.08 | 100 | 25.4% | 49.7% |
| I: H, mean pooling | 128 | 0.5 | 0.05 | mean | 0.4 / 0.35 / 0.08 | 100 | 31.6% | 57.9% |
| J: G, 256 units | 256 | 0.5 | 0.05 | mean | 0.3 / 0.25 / 0.05 | 100 | 33.9% | 56.2% |
| K: J, stronger aug | 256 | 0.5 | 0.05 | mean | 0.4 / 0.35 / 0.08 | 100 | 33.9% | 55.6% |
| L: J, 200 epochs | 256 | 0.5 | 0.05 | mean | 0.3 / 0.25 / 0.05 | 200 | 36.7% | 55.4% |
| **M: K, 200 epochs** | 256 | 0.5 | 0.05 | mean | 0.4 / 0.35 / 0.08 | 200 | **38.1%** | 57.3% |

What moved the needle, roughly in order of size: mean pooling over time instead of the last hidden
state (+5 to +6 points), a larger network once it is regularized (+5), augmentation that simulates
missed hands and shifted timing (+3 to +4), and longer training (+3 to +4). Config M is the default
of `signlang train`.

The first run (A) is the plain architecture from the project brief. With 671 training clips for 100
signs (about 7 per sign) it overfits quickly and loses to the baseline; the regularization and
pooling changes above are what close that gap.
