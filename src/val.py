"""Held-out validation: run a policy on the 20% validation split with the train-split checkpoints and score PQ."""
import sys, json, time, importlib.util, os, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
from pycocotools import mask as mu
from pycocotools.coco import COCO

ROOT = Path('data/MAGFiLO_1.0_Kaggle_2026')
def load_runner():
    spec = importlib.util.spec_from_file_location('r', str(Path(__file__).with_name('baseline_maskrcnn_refiner.py'))); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r); return r

def pq(gt, pred):
    """gt, pred: dict image -> list of RLE dicts. Organisers' formula (IoU > 0.5)."""
    s = 0.0; tp = fp = fn = 0
    for k in gt:
        g = gt[k]; p = pred.get(k, [])
        if not g: fp += len(p); continue
        if not p: fn += len(g); continue
        iou = np.asarray(mu.iou(p, g, [0] * len(g))).T  # (n_gt, n_pred)
        hit = iou > 0.5
        s += float(iou[hit].sum()); tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
    d = tp + 0.5 * fp + 0.5 * fn
    return dict(pq=s / d if d else 0.0, tp=tp, fp=fp, fn=fn, sq=s / tp if tp else 0.0, rq=tp / d if d else 0.0)

def load_gt(split='val'):
    man = pd.read_csv('weights/maskrcnn_validation/training_manifest.csv', dtype={'id': str})
    man = man[man['split'] == split]
    coco = COCO(str(ROOT / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    ids = {str(i): i for i in coco.imgs}
    gt = {}; files = {}
    for row in man.itertuples():
        iid = ids[row.id]
        rles = []
        for a in coco.imgToAnns[iid]:
            m = coco.annToMask(a)
            if m.any(): rles.append(mu.encode(np.asfortranarray(m)))
        gt[row.observation_id] = rles; files[row.observation_id] = ROOT / 'train/train_images' / row.file_name
    return gt, files

def run(r, det, ref, files, dev, overrides):
    for k, v in overrides.items(): setattr(r, k, v)
    out = {}
    for k, f in files.items():
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        out[k] = r.infer_observation(det, ref, img, device=dev)
    return out

if __name__ == '__main__':
    dev = torch.device(os.environ.get('FIL_DEV', 'mps'))
    r = load_runner()
    d = r.load_checkpoint(Path('weights/maskrcnn_validation/best_detector.pt')); f = r.load_checkpoint(Path('weights/maskrcnn_validation/best_refiner.pt'))
    print('ckpt epochs', d.get('epoch'), f.get('epoch'), flush=True)
    det = r.build_detector().to(dev); ref = r.CropUNet().to(dev); det.load_state_dict(d['model']); ref.load_state_dict(f['model']); det.eval(); ref.eval()
    gt, files = load_gt()
    print('val images', len(gt), 'gt instances', sum(len(v) for v in gt.values()), flush=True)
    configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    for c in configs:
        t = time.time(); out = run(r, det, ref, files, dev, c)
        res = {name: pq(gt, {k: v[i] for k, v in out.items()}) for i, name in enumerate(['anchor', 'attached', 'candidate'])}
        print(json.dumps(dict(cfg=c, sec=round(time.time() - t), **{n: {k: round(v, 4) for k, v in x.items()} for n, x in res.items()})), flush=True)
