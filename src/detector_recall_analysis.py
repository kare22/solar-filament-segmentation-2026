import pickle, numpy as np, torch, cv2, json
from pathlib import Path
from pycocotools import mask as mu
import val  # run from src/
r = val.load_runner(); dev = torch.device('cpu')
d = r.load_checkpoint(Path('weights/maskrcnn_validation/best_detector.pt')); det = r.build_detector(); det.load_state_dict(d['model']); det.eval()
det.roi_heads.score_thresh = 0.05; det.roi_heads.detections_per_img = 200
gt, files = val.load_gt()
raw = {}
for i, (k, f) in enumerate(files.items()):
    img = cv2.imread(str(f), 0)
    t = torch.from_numpy(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1) / 255
    with torch.inference_mode(): o = det([t])[0]
    ms = []
    for m in o['masks'][:, 0].numpy():
        ms.append(mu.encode(np.asfortranarray(cv2.resize((m >= 0.5).astype(np.uint8), (2048, 2048), interpolation=cv2.INTER_NEAREST))))
    raw[k] = dict(scores=o['scores'].numpy(), rles=ms, boxes=o['boxes'].numpy())
    if i % 20 == 0: print(i, flush=True)
pickle.dump(raw, open('out/val_raw_det.pkl', 'wb'))
# recall ceiling by score threshold and GT size
rows = []
for k in gt:
    g = gt[k]; p = raw[k]
    if not g: continue
    areas = mu.area(g)
    iou = np.asarray(mu.iou(p['rles'], g, [0] * len(g))).T if len(p['rles']) else np.zeros((len(g), 0))
    for gi in range(len(g)):
        best = {}
        for th in (0.05, 0.3, 0.5, 0.7):
            sel = p['scores'] >= th
            best[th] = float(iou[gi, sel].max()) if sel.any() else 0.0
        rows.append(dict(area=int(areas[gi]), **{str(t): v for t, v in best.items()}))
import pandas as pd
df = pd.DataFrame(rows); df['bin'] = pd.cut(df.area, [0, 300, 600, 1200, 2500, 6000, 1e9])
for th in ('0.05', '0.3', '0.5', '0.7'):
    print('score>=', th, 'GT with raw-mask IoU>0.5:', round((df[th] > 0.5).mean(), 3), ' IoU>0.3:', round((df[th] > 0.3).mean(), 3), ' IoU==0:', round((df[th] == 0).mean(), 3))
print(df.groupby('bin', observed=True).agg(n=('area', 'size'), hit07=('0.7', lambda s: (s > 0.5).mean()), hit005=('0.05', lambda s: (s > 0.5).mean()), miss=('0.05', lambda s: (s == 0).mean())).round(3))
for th in (0.3, 0.5, 0.7, 0.9): print('n det >=', th, sum(int((raw[k]['scores'] >= th).sum()) for k in raw))
