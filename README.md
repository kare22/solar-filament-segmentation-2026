# Solar Filament Segmentation Challenge 2026 — team Paan

Entry for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(MAGFiLO 1.0, GONG H-alpha full-disk images). Each row of the submission is one filament as a compressed COCO RLE mask
of a 2048 x 2048 observation.

This repository was put together on the last day of the competition. It does not contain a model we trained
ourselves. It contains (1) a reproducible local runner for two public pipelines, with credit to their authors,
(2) a held-out validation harness that uses the organisers' Panoptic Quality formula, and (3) two findings we think
are useful to the organisers: how the public pipelines rank under the current metric, and a robustness problem in
the published scoring formula.

## Results

| Pipeline | Held-out PQ (ours) | Public leaderboard |
|---|---|---|
| Mask R-CNN + crop U-Net refiner, flip TTA (public weights by Nomannic / Fususu) | 0.420 (142 held-out observations) | 0.34 |
| SOLFIRE `fast` preset (public weights by Vedran Hrabar), run locally | not measured by us (author reports 0.464 out-of-fold) | see `docs/RESULTS.md` |

The held-out number uses the 80/20 observation split and the train-split checkpoints published with the
Mask R-CNN notebook, so the 142 observations were not seen by the model that is scored on them.

## What is in here

| File | Purpose |
|---|---|
| `src/solfire_lean.py` | Inference-only extract of the SOLFIRE notebook (tiled ensemble, crop refiner, barb growth). Device is selectable, so it runs on Apple MPS or CPU as well as CUDA. |
| `src/solfire_run.py` | Runs the ensemble over a folder of observations, caches per-observation outputs, writes `submission.csv`. |
| `src/baseline_maskrcnn_refiner.py` | Local port of the public Mask R-CNN + refiner inference notebook (paths and device made configurable). |
| `src/val.py` | Held-out validation: runs a pipeline on the validation split and scores it with the organisers' PQ formula (IoU > 0.5). |
| `src/detector_recall_analysis.py` | How many ground-truth filaments the detector reaches at each score threshold, by filament size. |
| `src/metric_check.py` | Robustness check of the scoring formula (see below). |

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision opencv-python-headless pycocotools pandas numpy \
    segmentation-models-pytorch==0.6.0 scikit-image scipy tqdm
# competition data under data/MAGFiLO_1.0_Kaggle_2026 (kaggle competitions download filament-segmentation-2026)
# SOLFIRE weights: git clone https://github.com/vhrabar/solfire ext/solfire
SOLFIRE_PRESET=fast FIL_DEV=mps python src/solfire_run.py data/MAGFiLO_1.0_Kaggle_2026/test/test_images out/sf_test
```

On an Apple M1 Pro the `fast` preset takes about 40 s per observation (about two hours for the 180 test images).

## Findings

### 1. Thresholds do not rescue the detector route
On the held-out split the Mask R-CNN + refiner pipeline scores PQ 0.420 at its published settings
(664 true positives, 365 false positives, 450 false negatives). Moving the detector score threshold to 0.5 or 0.85,
or the refiner threshold to 0.5, changes PQ by less than 0.005 or lowers it: a lower threshold adds about 250 false
positives for 9 more true positives. The detector reaches only 60% of the annotated filaments with a mask of
IoU > 0.5 at its working threshold, and 15% of filaments under 300 px. The limit of this route is recall on small
filaments, not post-processing.

### 2. The published PQ formula rewards repeated masks
The self-evaluation notebook counts every (ground truth, prediction) pair with IoU > 0.5 as a true positive, so a
prediction that appears k times is counted k times. On our held-out split, with the same detector masks:

| Predictions | PQ |
|---|---|
| each mask once | 0.378 |
| masks with score >= 0.90 repeated 10 times | 0.519 |
| masks with score >= 0.95 repeated 50 times | 0.577 |

The score moves toward the mean IoU of the matched pairs (about 0.65) without any change in what is segmented.
`src/metric_check.py` reproduces this. We report it so that the final evaluation can match each ground-truth
filament to at most one prediction; **our submission contains every mask exactly once.**

## Credit and licences

- SOLFIRE (Vedran Hrabar, https://github.com/vhrabar/solfire): code under the MIT licence (`docs/LICENSE.solfire`),
  weights under CC BY-NC 4.0 (`docs/LICENSE-weights.solfire`). `src/solfire_lean.py` is an extract of that code.
  The weights are not redistributed here.
- Mask R-CNN + U-Net refiner: training notebook by Nomannic
  (https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner); inference notebook and
  flip-TTA by Fususu (https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff).
- Scoring formula: the organisers' self-evaluation notebook
  (https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
- Data: MAGFiLO 1.0 (doi:10.1038/s41597-024-03876-y), GONG H-alpha observations, NSO.

Our own code in this repository is released under the MIT licence.

## Validation checkpoints

`src/val.py` expects the train-split checkpoints and the split manifest published as output of the Mask R-CNN
training notebook (`best_detector.pt`, `best_refiner.pt`, `training_manifest.csv`) under
`weights/maskrcnn_validation/`.
