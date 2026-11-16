import csv
import hashlib
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TRAIN = False  # True: train the models and the crop refiners from the competition data; False: fetch the published ones
SEED = 1
FOLDS = 5  # grouped by observation, the assignment is fixed (seed 0) whatever SEED is

# Kaggle mode when the Kaggle mounts exist, local mode otherwise
KAGGLE = Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
SSL = True  # with TRAIN: start the ResNet-34 family from masked-image pretraining on the GONG H-alpha frames
SSL_EPOCHS = 30
SSL_PATCH, SSL_MASK_RATIO = 32, 0.6  # side of a hidden square (px) and the share of squares hidden

PRESET = os.environ.get("SOLFIRE_PRESET", "fast")  # final, fast, lite or tiny
MODEL_HANDLE = f"vhrabar/filament-segmentation-2026/pyTorch/solfire-{PRESET}/1"  # Kaggle Models variation
CALIBRATION_NAME = f"{PRESET}.calibration.json"  # cut-offs and inference settings, next to the checkpoints
WEIGHTS_DIR = None  # an override folder holding the checkpoints and the calibration
# the repository checkout this runs in (the notebook sits in notebook/), else the working directory
REPO = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / "notebook" / "pipeline.py").is_file()), Path.cwd())
REPO_WEIGHTS = REPO / "weights" / "final"

# %%
FAMILIES = {
    "unetpp-r34": {"decoder": "unetplusplus", "encoder": "resnet34"},
    "unet-mitb2": {"decoder": "unet", "encoder": "mit_b2"},
    "unet-hrnet32": {"decoder": "unet", "encoder": "tu-hrnet_w32"},
}
TRAINING = {  # shared by both stages
    "crop": 512,  # window edge in native pixels
    "crops_per_image": 7,  # windows per observation and epoch
    "foreground_prob": 0.7,  # share of windows placed on an annotated filament
    "batch_size": 8,
    "weight_decay": 1e-2,
    "encoder_lr_scale": 0.25,  # the pretrained encoder learns at a quarter of the decoder's rate
    "pos_weight": 5.0,
    "dice_weight": 0.5,  # BCE gets the rest
    "grad_clip": 1.0,
    "val_limit": 64,  # held-out records scored after every epoch to pick the best one
}
STAGES = {
    "stage1": {"epochs": 40, "lr": 4e-4, "warmup_epochs": 3, "lovasz_weight": 0.0},
    "stage2": {"epochs": 8, "lr": 1e-4, "warmup_epochs": 0, "lovasz_weight": 0.5},
}
TILE, OVERLAP, TILE_BATCH = 512, 0.25, 8
CALIBRATION_GRID = {
    "threshold": (0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.65),
    "offset": (-2, -1, 0, 1),  # signed one-pixel dilation/erosion steps
    "min_area": (0, 128, 256, 512),
    "min_score": (0.0, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 0.85, 0.88, 0.91, 0.94),
}

# %%
REFINER = {
    "crop": 256,  # edge the refiner works at
    "context": 1.5,  # window edge as a multiple of the instance's longer box side
    "min_window": 128,
    "min_pixels": 32,  # smaller proposals are never refined nor submitted
    "proposal_thresholds": (0.3, 0.375, 0.45),  # training proposals; the calibrated cut sits in the middle
    "duplicate_iou": 0.85,  # a training proposal this close to one already taken is skipped
    "epochs": 12,
    "batch_size": 32,
    "lr": 3e-4,
    "encoder_lr_scale": 0.5,
    "score_weight": 0.5,
}
BARBS = None  # None follows the calibration's "barbs" block; True / False forces the barbs on (DEFAULT_BARBS) or off
REFINE_GRID = {
    "threshold": (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6),
    "min_score": tuple(round(0.02 * step, 2) for step in range(31)),  # cut on the predicted IoU
}

SUBMISSION_SIZE = 2048
WORKERS = min(8, os.cpu_count() or 1)
WORK = Path("/kaggle/working") if KAGGLE else REPO / "build" / "notebook"
WORK.mkdir(parents=True, exist_ok=True)

# %%
try:
    import segmentation_models_pytorch as smp
except ImportError:  # not in Kaggle's image
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0", "timm==1.0.30"], check=True
    )
    import segmentation_models_pytorch as smp

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import disk
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DEVICE = torch.device(os.environ.get("FIL_DEV","mps"))


def set_seeds(seed: int) -> None:
    """Seed every random source; called at the start of each training run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seeds(SEED)
print(f"torch {torch.__version__}, smp {smp.__version__}, device {DEVICE}, workers {WORKERS}")

import pandas as pd



def load_image(path, window=None) -> Tensor:
    """
    A greyscale observation, or one square window of it, as a (1, H, W) float tensor in [0, 1].
    :param path: Observation file
    :param window: (x, y, edge) of the window, or None for the whole observation
    :return: (1, H, W) tensor
    """
    with Image.open(path) as handle:
        image = handle.convert("L")
        if window is not None:
            x, y, edge = window
            image = image.crop((x, y, x + edge, y + edge))
        array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).unsqueeze(0).float().div_(255.0)


IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # averaged over the colour channels
DECODERS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}


class Pretrained(nn.Module):
    """
    smp decoder + ImageNet encoder for one-channel input and one-channel logits, at any input size.
    """

    def __init__(self, decoder: str, encoder: str, encoder_weights=None):
        """
        :param decoder: Key of DECODERS
        :param encoder: smp encoder name, tu- prefixed for timm encoders
        :param encoder_weights: "imagenet" to start training, None when a checkpoint supplies the weights
        """
        super().__init__()
        self.net = DECODERS[decoder](encoder_name=encoder, encoder_weights=encoder_weights, in_channels=1, classes=1)
        self.divisor = 32  # five halvings in the encoder

    @property
    def encoder(self) -> nn.Module:
        return self.net.encoder

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (B, 1, H, W) observations in [0, 1]
        :return: (B, 1, H, W) logits
        """
        height, width = x.shape[-2:]
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = F.pad(x, (0, -width % self.divisor, 0, -height % self.divisor), mode="replicate")
        return self.net(x)[..., :height, :width]


def load_model(path) -> tuple[nn.Module, dict]:
    """
    Rebuild a trained model from its checkpoint.
    :param path: Checkpoint file
    :return: Tuple of (model in eval mode on DEVICE, checkpoint without the weights)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = Pretrained(config["decoder"], config["encoder"])
    model.load_state_dict(checkpoint.pop("model_state"))
    return model.to(DEVICE).eval(), checkpoint

# %%
@torch.no_grad()
def predict_flips(model, images: Tensor, turned=False) -> Tensor:
    """
    :param model: Model in eval mode
    :param images: (B, 1, H, W) windows on DEVICE
    :param turned: Use the four flips of the quarter-turned windows instead (the other half of the square's symmetries)
    :return: (B, 1, H, W) probabilities averaged over the identity and the three flips
    """
    images = torch.rot90(images, 1, (-2, -1)) if turned else images
    total = torch.zeros_like(images, dtype=torch.float32)
    for flip_x in (False, True):
        for flip_y in (False, True):
            dims = [d for d, on in ((-1, flip_x), (-2, flip_y)) if on]
            view = torch.flip(images, dims) if dims else images
            probabilities = torch.sigmoid(model(view).float())
            total += torch.flip(probabilities, dims) if dims else probabilities
    return torch.rot90(total / 4, -1, (-2, -1)) if turned else total / 4

# %%
def tile_offsets(size: int, tile: int, stride: int) -> list[int]:
    """
    :return: Window starts along one axis; the last window is pulled back to the edge rather than padded
    """
    if size <= tile:
        return [0]
    offsets = list(range(0, size - tile + 1, stride))
    if offsets[-1] != size - tile:
        offsets.append(size - tile)
    return offsets


@torch.no_grad()
def predict_tiled(
    model, image: Tensor, tile=TILE, overlap=OVERLAP, tta=True, batch_size=TILE_BATCH, scale=1.0, turned=False, skip_dark=False
) -> Tensor:
    """
    Probabilities for one whole observation, assembled from overlapping windows.
    :param model: Model in eval mode
    :param image: (1, H, W) observation at native resolution
    :param tile: Window edge
    :param overlap: Fraction of a window shared with its neighbour
    :param tta: Average every window over the four flips
    :param batch_size: Windows per forward pass
    :param scale: Resize the observation by this factor first and the probabilities back afterwards
        (2 for the zoom family)
    :param turned: Average over the flips of the quarter-turned windows instead (see predict_flips)
    :param skip_dark: Give windows with nothing brighter than the black off the disk probability 0 without running
        the model on them
    :return: (H, W) probabilities on the cpu
    """
    if scale != 1.0:
        height, width = image.shape[-2:]
        size = (round(height * scale), round(width * scale))
        resized = F.interpolate(image[None].float(), size=size, mode="bilinear", align_corners=False, antialias=scale < 1.0)
        probabilities = predict_tiled(model, resized[0], tile, overlap, tta, batch_size, 1.0, turned, skip_dark)
        return F.interpolate(probabilities[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    image = image.unsqueeze(0)
    height, width = image.shape[-2:]
    stride = max(1, round(tile * (1.0 - overlap)))
    corners = [(y, x) for y in tile_offsets(height, tile, stride) for x in tile_offsets(width, tile, stride)]

    ramp = torch.hann_window(tile + 2, periodic=False, device=DEVICE)[1:-1].clamp_min(0.05)
    weights = (ramp[:, None] * ramp[None, :])[None, None]
    total = torch.zeros((1, 1, height, width), dtype=torch.float32, device=DEVICE)
    counts = torch.zeros_like(total)
    if skip_dark:
        lit = [(y, x) for y, x in corners if image[..., y : y + tile, x : x + tile].max() > 10 / 255]
        for y, x in set(corners) - set(lit):
            counts[..., y : y + tile, x : x + tile] += weights
        corners = lit
    for start in range(0, len(corners), batch_size):
        chunk = corners[start : start + batch_size]
        batch = torch.cat([image[..., y : y + tile, x : x + tile] for y, x in chunk]).to(DEVICE, non_blocking=True)
        probabilities = predict_flips(model, batch, turned) if tta else torch.sigmoid(model(batch).float())
        for (y, x), probability in zip(chunk, probabilities):
            total[..., y : y + tile, x : x + tile] += probability[None] * weights
            counts[..., y : y + tile, x : x + tile] += weights
    return (total / counts)[0, 0].cpu()

# %%
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n_labels = ndimage.label(mask, structure=EIGHT_CONNECTED)
    return labels.astype(np.int32, copy=False), n_labels


def apply_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    """
    Dilate (offset > 0) or erode (offset < 0) by |offset| steps of the four-neighbour diamond.
    """
    grow = offset > 0
    mask = mask if grow else ~mask
    for _ in range(abs(offset)):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask if grow else ~mask


def component_stats(labels: np.ndarray, n_labels: int, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    :return: Tuple of (area, mean probability) per component, component k at index k - 1
    """
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n_labels + 1)[1:]
    sums = np.bincount(flat, weights=probabilities.ravel(), minlength=n_labels + 1)[1:]
    return areas, sums / np.maximum(areas, 1)


def label_instances(probabilities: np.ndarray, threshold, offset=0, min_area=0, min_score=0.0) -> tuple[np.ndarray, int]:
    """
    Turn a probability map into numbered instances, the way the submission is written.
    :return: Tuple of ((H, W) labels with 0 as background, number of instances)
    """
    labels, n_labels = connected_components(apply_offset(probabilities >= threshold, offset))
    areas, scores = component_stats(labels, n_labels, probabilities)
    keep = (areas > min_area) & (scores >= min_score)
    remap = np.zeros(n_labels + 1, dtype=np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels], int(keep.sum())

# %%
def annotation_pixels(annotations) -> list[np.ndarray]:
    """
    :param annotations: The filaments of one annotator record
    :return: Flat pixel indices of each filament, filled by pycocotools like the scored masks
    """
    pixels = []
    for annotation in annotations:
        rle = coco_mask.merge(coco_mask.frPyObjects(annotation["segmentation"], SUBMISSION_SIZE, SUBMISSION_SIZE))
        pixels.append(np.flatnonzero(coco_mask.decode(rle)))
    return pixels


def instance_iou(gt_pixels, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """
    :return: (n_gt, n_labels) IoU of every annotated filament with every predicted component
    """
    flat = labels.ravel()
    pred_areas = np.bincount(flat, minlength=n_labels + 1)[1:]
    iou = np.zeros((len(gt_pixels), n_labels))
    for row, pixels in enumerate(gt_pixels):
        intersection = np.bincount(flat[pixels], minlength=n_labels + 1)[1:]
        iou[row] = intersection / np.maximum(len(pixels) + pred_areas - intersection, 1)
    return iou


def pq_terms(iou: np.ndarray) -> np.ndarray:
    """
    :param iou: (n_gt, n_pred) IoU matrix of one record
    :return: [sum of matched IoU, TP, FP, FN]
    """
    hits = iou > 0.5
    return np.array([iou[hits].sum(), hits.sum(), (~hits.any(axis=0)).sum(), (~hits.any(axis=1)).sum()], dtype=np.float64)


def panoptic_quality(terms) -> float:
    matched, tp, fp, fn = terms
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(matched / denominator) if denominator else 0.0


def component_window(box, context=REFINER["context"]) -> tuple[int, int, int]:
    """
    :param box: (rows, columns) slices of a component, from ndimage.find_objects
    :param context: Window edge as a multiple of the longer box side
    :return: (top, left, side) of the square window it is refined on; it may reach past the image
    """
    rows, columns = box
    side = max(REFINER["min_window"], round(context * max(rows.stop - rows.start, columns.stop - columns.start)))
    return round((rows.start + rows.stop - side) / 2), round((columns.start + columns.stop - side) / 2), side


def clipped(window, height: int, width: int) -> tuple[slice, slice, slice, slice]:
    """:return: (rows, columns) of the window inside the observation, then the same region inside the window"""
    top, left, side = window
    y0, x0, y1, x1 = max(top, 0), max(left, 0), min(top + side, height), min(left + side, width)
    return slice(y0, y1), slice(x0, x1), slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)


def window_of(array: np.ndarray, window) -> np.ndarray:
    """:return: (side, side) cut of the array, zeros past its edges"""
    out = np.zeros((window[2], window[2]), dtype=array.dtype)
    rows, columns, window_rows, window_columns = clipped(window, *array.shape)
    out[window_rows, window_columns] = array[rows, columns]
    return out


def resample(windows: Tensor, size: int) -> Tensor:
    """(B, C, s, s) -> (B, C, size, size), antialiased on the way down"""
    if windows.shape[-1] == size:
        return windows
    return F.interpolate(windows, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def refiner_inputs(image: np.ndarray, probabilities: np.ndarray, seed: np.ndarray, window) -> Tensor:
    """:return: (3, crop, crop) observation, ensemble probability and seed mask, in [0, 1]"""
    stacked = np.stack([window_of(image, window), window_of(probabilities, window), window_of(seed, window).astype(np.float32)])
    return resample(torch.from_numpy(stacked)[None], REFINER["crop"])[0].clamp_(0, 1)

# %%
class Refiner(nn.Module):
    """ResNet-34 U-Net that redraws one instance, with a head on the bottleneck predicting the IoU it will earn."""

    def __init__(self, encoder_weights=None):
        super().__init__()
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            aux_params={"classes": 1, "dropout": 0.2},
        )

    @property
    def encoder(self) -> nn.Module:
        return self.net.encoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """:return: Tuple of ((B, 1, S, S) mask logits, (B,) logits of the earned IoU)"""
        masks, scores = self.net((x - IMAGENET_MEAN) / IMAGENET_STD)
        return masks, scores[:, 0]


def load_refiner(path) -> Refiner:
    model = Refiner()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict_refiner(refiners, inputs: Tensor) -> tuple[Tensor, Tensor]:
    """
    :return: Mask probabilities and predicted IoU, averaged over the refiners and three flips; stacked and averaged
        at once (under autocast a running float16 sum rounds differently from the solfire package)
    """
    masks, scores = [], []
    for model in refiners:
        for dims in ((), (-1,), (-2,)):
            logits, score = model(torch.flip(inputs, dims) if dims else inputs)
            masks.append(torch.sigmoid(torch.flip(logits, dims) if dims else logits))
            scores.append(torch.sigmoid(score))
    return torch.stack(masks).mean(0).float(), torch.stack(scores).mean(0).float()

# %%
def refine_components(
    refiners, image: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, n_labels: int, context=REFINER["context"]
) -> list:
    """
    :param context: Window edge as a multiple of the longer box side, as the refiners were trained
    :return: Per proposal of at least min_pixels pixels, (window, uint8 refined probabilities at native size, score)
    """
    areas = np.bincount(labels.ravel(), minlength=n_labels + 1)
    jobs = [
        (index + 1, component_window(box, context))
        for index, box in enumerate(ndimage.find_objects(labels))
        if box and areas[index + 1] >= REFINER["min_pixels"]
    ]
    refined = []
    for start in range(0, len(jobs), REFINER["batch_size"]):
        chunk = jobs[start : start + REFINER["batch_size"]]
        inputs = torch.stack([refiner_inputs(image, probabilities, labels == k, window) for k, window in chunk])
        masks, scores = predict_refiner(refiners, inputs.to(DEVICE))
        for (_, window), mask, score in zip(chunk, masks, scores):
            native = resample(mask[None].float(), window[2])[0, 0]
            refined.append((window, (native * 255).round().byte().cpu().numpy(), float(score)))
    return refined


def assemble(refined, threshold: float, min_score=0.0) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Paint refined instances into one label array, most confident first; a mask that more than half falls on
    instances already painted is a duplicate and is dropped.
    :return: Tuple of ((H, W) labels, number of instances, their scores)
    """
    labels = np.zeros((SUBMISSION_SIZE, SUBMISSION_SIZE), dtype=np.int32)
    scores = []
    for window, probabilities, score in sorted(refined, key=lambda item: -item[2]):
        if score < min_score:
            break
        rows, columns, window_rows, window_columns = clipped(window, SUBMISSION_SIZE, SUBMISSION_SIZE)
        components, count = connected_components(probabilities[window_rows, window_columns] >= round(threshold * 255))
        mask = components == int(np.bincount(components.ravel())[1:].argmax()) + 1 if count > 1 else components > 0
        area = int(mask.sum())
        free = mask & (labels[rows, columns] == 0)
        if area < REFINER["min_pixels"] or free.sum() < 0.5 * area:
            continue
        scores.append(score)
        labels[rows, columns][free] = len(scores)
    return labels, len(scores), np.asarray(scores)


def refine_observation(refiners, image: np.ndarray, probabilities: np.ndarray, settings: dict) -> tuple[np.ndarray, int, dict]:
    """
    The instances of one observation the way the refined submission is written.
    :return: Tuple of (labels, number of instances, details): the proposals the refiner started from and the
        predicted IoU of the kept instances and of every candidate (the ones under the cut are dropped)
    """
    proposals = apply_offset(probabilities >= settings["proposal_threshold"], settings["proposal_offset"])
    proposed, n_proposed = connected_components(proposals)
    context = settings.get("context", REFINER["context"])
    refined = refine_components(refiners, image, probabilities, proposed, n_proposed, context)
    labels, n_labels, scores = assemble(refined, settings["threshold"], settings["min_score"])
    details = {"proposals": proposed, "kept": scores, "candidates": np.array([item[2] for item in refined])}
    return labels, n_labels, details



# %%
# contrast and ridge cut the candidates; a piece is kept when length / mean width >= min_ratio and width <= max_width
DEFAULT_BARBS = {"contrast": 0.05, "ridge": 0.025, "band": 20, "min_ratio": 4.0, "max_width": 4.0}


def local_contrast(image: np.ndarray) -> np.ndarray:
    """(background - image) / background, smoothed; the background is a median taken on a 4x smaller copy."""
    height, width = image.shape
    rows, columns = -(-height // 4) * 4, -(-width // 4) * 4
    padded = np.pad(image.astype(np.float32), ((0, rows - height), (0, columns - width)), mode="edge")
    small = padded.reshape(rows // 4, 4, columns // 4, 4).mean((1, 3))
    background = np.kron(ndimage.median_filter(small, 15), np.ones((4, 4), np.float32))[:height, :width]
    contrast = np.where(background > 20 / 255, (background - image) / np.maximum(background, 1e-6), 0)
    return ndimage.gaussian_filter(contrast.astype(np.float32), 1)


def thin_pieces(pieces: np.ndarray, mask: np.ndarray, min_ratio: float, max_width: float) -> np.ndarray:
    """The long, narrow pieces, measured from where they leave the mask."""
    labels, n_labels = ndimage.label(pieces, structure=EIGHT_CONNECTED)
    distance = ndimage.distance_transform_edt(~mask)
    keep = np.zeros(n_labels + 1, dtype=bool)
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        points = np.argwhere(labels[window] == number) + np.array([window[0].start, window[1].start])
        base = points[np.argmin(distance[points[:, 0], points[:, 1]])]
        length = np.sqrt(((points - base) ** 2).sum(1).max()) + 1
        width = len(points) / length
        keep[number] = length / width >= min_ratio and width <= max_width
    return keep[labels]


def grow_barbs(labels: np.ndarray, n_labels: int, image: np.ndarray, settings: dict) -> np.ndarray:
    """Add the visible barbs of every instance to it; grown pixels only fill background."""
    settings = {**DEFAULT_BARBS, **settings}
    if n_labels == 0:
        return labels
    contrast = local_contrast(image)
    grown = labels.copy()
    margin = settings["band"] + 2
    height, width = labels.shape
    for number, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        window = (
            slice(max(window[0].start - margin, 0), min(window[0].stop + margin, height)),
            slice(max(window[1].start - margin, 0), min(window[1].stop + margin, width)),
        )
        mask = labels[window] == number
        local = contrast[window]
        ridge = sato(local, sigmas=[1, 2], black_ridges=False)
        band = ndimage.binary_dilation(mask, disk(int(settings["band"]))) & ~mask
        candidates = band & (local > settings["contrast"]) & (ridge > settings["ridge"])
        pieces, _ = ndimage.label(candidates, structure=EIGHT_CONNECTED)
        touching = np.unique(pieces[ndimage.binary_dilation(mask, EIGHT_CONNECTED) & candidates])
        barbs = thin_pieces(np.isin(pieces, touching[touching > 0]), mask, settings["min_ratio"], settings["max_width"])
        target = grown[window]
        target[barbs & (target == 0)] = number
    return grown


WEIGHTS=Path(os.environ.get("SOLFIRE_WEIGHTS","ext/solfire/weights/final"))
calibration=json.loads((WEIGHTS/CALIBRATION_NAME).read_text())
checkpoints=[WEIGHTS/Path(p).name for p in calibration["checkpoints"]]
calibration["refiner"]["checkpoints"]=[str(WEIGHTS/Path(p).name) for p in calibration["refiner"]["checkpoints"]]
REFINE=True; refiner_settings=calibration["refiner"]


def encode_rle(mask: np.ndarray) -> str:
    """Compressed COCO RLE counts of a 2048 x 2048 mask, as the submission expects them."""
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def parse_variant(name: str) -> dict:
    """Inference variant of a calibration: base, or '-'-joined o<overlap> and turned (e.g. o0.5, turned)."""
    settings = {"overlap": OVERLAP, "turned": False}
    for part in [] if name == "base" else name.split("-"):
        if part == "turned":
            settings["turned"] = True
        elif part.startswith("o"):
            settings["overlap"] = float(part[1:])
        else:
            raise ValueError(f"unknown part {part!r} of inference variant {name!r}")
    return settings

# %%
variants = [parse_variant(name) for name in calibration.get("variants") or ["base"]]
# float16 autocast when the calibration asks for it; the cached out-of-fold maps were float32, the masks match
half = calibration.get("precision") == "fp16" and DEVICE.type == "cuda"
skip_dark = bool(calibration.get("skip_dark"))
barbs = calibration.get("barbs") if BARBS is None else ((calibration.get("barbs") or {}) if BARBS else None)
print("barbs grown into the instances" if barbs is not None else "no barbs grown")
models, refiners = [], []


def load_inference() -> None:
    """Load the ensemble and the refiners on first use, so a finished prediction run does not need them."""
    if models:
        return
    for path in checkpoints:
        model, meta = load_model(path)
        models.append((model, meta.get("input_scale") or 1.0))  # the zoom family predicts at 2x
    if REFINE:
        refiners.extend(load_refiner(path) for path in refiner_settings["checkpoints"])
    print(f"{len(models)} models x {len(variants)} variants, {len(refiners)} refiners, {'fp16' if half else 'fp32'}")

# %%
def infer(image: Tensor) -> dict:
    """:return: The stages of one observation: probabilities, spread, proposals, refined, labels, n_labels, scores, candidates"""
    load_inference()
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=half):
        maps = [
            predict_tiled(model, image, scale=scale, skip_dark=skip_dark, **variant)
            for model, scale in models
            for variant in variants
        ]
        probabilities = (sum(maps) / len(maps)).numpy()
        refined = refine_observation(refiners, image.numpy()[0], probabilities, refiner_settings) if REFINE else None
    if refined is not None:
        labels, n_labels, details = refined
    else:
        labels, n_labels = label_instances(
            probabilities, calibration["threshold"], calibration["offset"], calibration["min_area"], calibration["min_score"]
        )
        mean = ndimage.mean(probabilities, labels, range(1, n_labels + 1)) if n_labels else []
        details = {"proposals": labels, "kept": np.asarray(mean, float), "candidates": np.asarray(mean, float)}
    per_model = [sum(maps[i : i + len(variants)]) / len(variants) for i in range(0, len(maps), len(variants))]
    result = {
        "probabilities": probabilities,
        "spread": torch.stack(per_model).std(dim=0).numpy(),  # standard deviation between the models
        "refined": labels,
        "n_labels": n_labels,
        **details,
    }
    if barbs is not None:
        labels = grow_barbs(labels, n_labels, image.numpy()[0], barbs)
    result["labels"] = labels
    return result

# %%
SHOW = 2
RERUN = False  # True: predict again even when a finished prediction of this PRESET is in WORK
SUBMISSION = WORK / "submission.csv"
SUMMARY = WORK / "inference.json"  # what the figures need from a finished prediction, next to the submission
RUN_TAG = "trained" if TRAIN else PRESET
examples, extremes, summary = [], {}, []


