"""
One-time S3DIS preprocessing: raw .txt annotation files → per-room .npy arrays.

Reads the Stanford 3D Indoor Spaces dataset
    data/Stanford3dDataset_v1.2_Aligned_Version/Area_i/<room>/Annotations/*.txt
and produces
    data/s3dis_npy/Area_i_<room>.npy           # [N, 7] = x y z r g b label

The standard S3DIS protocol uses 13 semantic classes. The `stairs` category
(which only appears in a handful of rooms) is mapped to `clutter`, matching
the PointNet/PointNet++/DGCNN convention.

Canonical 13-class ordering (used by every public S3DIS benchmark):
    0: ceiling   1: floor     2: wall       3: beam      4: column
    5: window    6: door      7: table      8: chair     9: sofa
   10: bookcase  11: board   12: clutter

Run from the LIDARLearn project root:
    python preprocessing/preprocess_s3dis.py
"""

import glob
import os
from pathlib import Path

import numpy as np

# Standard 13-class label map. Any annotation file stem starting with one of
# these names maps to the corresponding integer label.
S3DIS_CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter',
]
LABEL_MAP = {name: i for i, name in enumerate(S3DIS_CLASSES)}
# Map 'stairs' → 'clutter' (standard convention)
LABEL_MAP['stairs'] = LABEL_MAP['clutter']


def parse_points_txt(path: str) -> np.ndarray:
    """Load an S3DIS .txt file of lines 'x y z r g b'. Returns [N, 6] float32.

    A few files contain non-numeric tokens (stray characters from the raw
    Matterport export). We skip those lines.
    """
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                rows.append([float(x) for x in parts[:6]])
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def process_room(room_dir: Path) -> np.ndarray:
    """Merge all Annotations/*.txt files in a room → [N, 7] = xyz rgb label."""
    ann_dir = room_dir / 'Annotations'
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"No Annotations/ in {room_dir}")

    chunks = []
    for txt_path in sorted(ann_dir.glob('*.txt')):
        # Category from the filename stem ("chair_5" → "chair").
        stem = txt_path.stem
        cat = stem.rsplit('_', 1)[0]
        if cat not in LABEL_MAP:
            # Unknown category → treat as clutter.
            label = LABEL_MAP['clutter']
        else:
            label = LABEL_MAP[cat]
        pts = parse_points_txt(str(txt_path))
        if pts.size == 0:
            continue
        labels = np.full((pts.shape[0], 1), label, dtype=np.float32)
        chunks.append(np.concatenate([pts, labels], axis=1))

    if not chunks:
        return np.zeros((0, 7), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def main():
    root = Path(__file__).resolve().parent.parent          # LIDARLearn/
    src_root = root / 'data' / 'Stanford3dDataset_v1.2_Aligned_Version'
    dst_root = root / 'data' / 's3dis_npy'
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.is_dir():
        raise FileNotFoundError(f"S3DIS source not found: {src_root}")

    area_dirs = sorted(p for p in src_root.glob('Area_*') if p.is_dir())
    total_rooms = 0
    for area_dir in area_dirs:
        area_name = area_dir.name  # "Area_1"
        room_dirs = sorted(p for p in area_dir.iterdir() if p.is_dir())
        for room_dir in room_dirs:
            room_name = room_dir.name
            out_path = dst_root / f"{area_name}_{room_name}.npy"
            if out_path.exists():
                total_rooms += 1
                continue
            try:
                data = process_room(room_dir)
            except Exception as e:
                print(f"[skip] {area_name}/{room_name}: {e}")
                continue
            if data.size == 0:
                print(f"[skip] {area_name}/{room_name}: empty")
                continue
            np.save(str(out_path), data)
            total_rooms += 1
            print(f"  {out_path.name}: {data.shape[0]:>8d} pts")

    print(f"\nDone. {total_rooms} rooms written to {dst_root}")


if __name__ == '__main__':
    main()
