"""Robustness check of the published Panoptic Quality formula on held-out training images.

The self-evaluation notebook counts every (ground truth, prediction) pair with IoU > 0.5 as a true positive.
A prediction that is repeated k times is therefore counted k times. This script measures, on the held-out
validation split only, how much the score moves when confident masks are repeated. It is a report for the
organisers; no repeated masks are used in our submission.
"""
import pickle
import val

def repeated(raw, threshold, repeat_threshold=None, k=1):
    out = {}
    for key, p in raw.items():
        rows = []
        for score, rle in zip(p['scores'], p['rles']):
            if score >= threshold:
                rows += [rle] * (k if repeat_threshold is not None and score >= repeat_threshold else 1)
        out[key] = rows
    return out

if __name__ == '__main__':
    gt, _ = val.load_gt()
    raw = pickle.load(open('out/val_raw_det.pkl', 'rb'))  # written by detector_recall_analysis.py
    for name, args in (('each mask once', (0.7,)), ('masks >= 0.90 repeated x10', (0.7, 0.9, 10)), ('masks >= 0.95 repeated x50', (0.7, 0.95, 50))):
        print(name, {k: round(v, 3) for k, v in val.pq(gt, repeated(raw, *args)).items()})
