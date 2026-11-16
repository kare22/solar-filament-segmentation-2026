"""Run the SOLFIRE ensemble (vhrabar/solfire, public weights) on a folder of observations; cache probability maps and instances."""
import sys, os, time, pickle, csv
from pathlib import Path
import numpy as np
import solfire_lean as S
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
files = sorted(p for p in src.iterdir() if p.suffix.lower() in {'.jpeg', '.jpg', '.png'})[:limit]
rows = []; t0 = time.time()
for i, p in enumerate(files):
    cache = out / f'{p.stem}.npz'
    if cache.exists():
        z = np.load(cache); labels, n = z['labels'], int(z['n'])
    else:
        r = S.infer(S.load_image(p)); labels, n = r['labels'], int(r['n_labels'])
        np.savez_compressed(cache, labels=labels.astype(np.uint16), refined=r['refined'].astype(np.uint16), n=n, prob=(r['probabilities'] * 255).astype(np.uint8), kept=np.asarray(r['kept'], np.float32), candidates=np.asarray(r['candidates'], np.float32))
    rows += [(f'{p.stem}_{l}', S.encode_rle(labels == l)) for l in range(1, n + 1) if (labels == l).any()]
    print(i + 1, len(files), p.stem, n, round(time.time() - t0), flush=True)
with open(out / 'submission.csv', 'w', newline='') as h:
    w = csv.writer(h, quoting=csv.QUOTE_NONE, quotechar=None); w.writerow(['filament_id', 'segmentation_rle']); w.writerows(rows)
print('rows', len(rows), 'sec', round(time.time() - t0))
