import argparse
import csv
import json
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.swinmatcher import SwinMatcher, default_cfg

warnings.filterwarnings("ignore")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SwinMatcher using the paper protocol: NCM, Pre, SR and RMSE "
            "computed directly from ground-truth transformation labels without RANSAC filtering."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/disk1/Data/datasets/SwinMatcher/BIT_multi-modal_datasets_homography_180_rotation",
        help="Dataset root with scene/scene_pair/folder structure.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="manifests/test_SwinMatcherBIT_gt.jsonl",
        help="JSONL manifest to evaluate. When set, this takes precedence over --data_root.",
    )
    parser.add_argument("--ckpt_path", type=str, default="weights/swinmatcher_512.ckpt")
    parser.add_argument("--output_dir", type=str, default="outputs_eval")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--coarse_thr", type=float, default=0.3, help="Paper setting theta_c.")
    parser.add_argument("--fine_thr", type=float, default=0.1, help="Paper setting theta_f.")
    parser.add_argument("--correct_thr", type=float, default=5.0, help="Correct match threshold in pixels.")
    parser.add_argument("--success_ncm", type=int, default=20, help="A pair succeeds if NCM >= this value.")
    parser.add_argument("--failed_rmse", type=float, default=10.0, help="RMSE assigned to failed pairs.")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--pair_stride", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--half", action="store_true", help="Use fp16 inference on CUDA.")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--vis_limit", type=int, default=50)
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def modality_pair_name(scene, scene_pair):
    prefix = f"{scene}_"
    if scene_pair.startswith(prefix):
        return scene_pair[len(prefix):]
    return scene_pair


def build_pairs(root_dir):
    pairs = []
    skipped_without_matrix = 0
    root_dir = Path(root_dir)

    for scene_path in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for scene_pair_path in sorted(p for p in scene_path.iterdir() if p.is_dir()):
            for folder_path in sorted(p for p in scene_pair_path.iterdir() if p.is_dir()):
                files = sorted(p.name for p in folder_path.iterdir() if p.is_file())
                images = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
                if len(images) < 2:
                    continue

                image0 = images[0]
                candidates = images[2:] if len(images) > 2 else images[1:]
                for image1 in candidates:
                    matrix_path = folder_path / f"{Path(image1).stem}.txt"
                    if not matrix_path.exists():
                        skipped_without_matrix += 1
                        continue
                    pairs.append(
                        {
                            "scene": scene_path.name,
                            "scene_pair": scene_pair_path.name,
                            "modality_pair": modality_pair_name(scene_path.name, scene_pair_path.name),
                            "folder": folder_path.name,
                            "image0_path": str(folder_path / image0),
                            "image1_path": str(folder_path / image1),
                            "matrix_path": str(matrix_path),
                        }
                    )

    return pairs, skipped_without_matrix


def load_matrix(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    else:
        matrix = np.loadtxt(path)
    return matrix.astype(np.float32)


def load_manifest_pairs(manifest_path):
    pairs = []
    skipped = 0
    manifest_path = Path(manifest_path)

    with manifest_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("mode") != "gt_pairs":
                continue

            image0_path = record.get("image0")
            image1_path = record.get("image1")
            matrix_path = record.get("gt")
            if not image0_path or not image1_path or not matrix_path:
                skipped += 1
                continue
            if not (Path(image0_path).is_file() and Path(image1_path).is_file() and Path(matrix_path).is_file()):
                skipped += 1
                continue

            modality_pair = f"{record.get('modality0', 'unknown')}-{record.get('modality1', 'unknown')}"
            pairs.append(
                {
                    "id": record.get("id", ""),
                    "scene": record.get("scene", ""),
                    "scene_pair": record.get("subset", ""),
                    "modality_pair": modality_pair,
                    "folder": Path(image0_path).parent.name,
                    "image0_path": image0_path,
                    "image1_path": image1_path,
                    "matrix_path": matrix_path,
                    "gt_direction": record.get("gt_direction", "0to1"),
                }
            )

    return pairs, skipped


def prepare_pairs(pairs, args):
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)
    if args.pair_stride > 1:
        pairs = pairs[:: args.pair_stride]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    return pairs


def load_model(args, device):
    config = default_cfg
    config["match_coarse"]["thr"] = args.coarse_thr
    config["match_coarse"]["border_rm"] = 2
    config["fine"]["thr"] = args.fine_thr
    config["img_size"] = args.img_size

    model = SwinMatcher(config=config)
    ckpt = load_checkpoint(args.ckpt_path, device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    if args.half and device.type == "cuda":
        model = model.half()

    return model


def read_pair_images(pair, img_size):
    image0_bgr = cv2.imread(pair["image0_path"], cv2.IMREAD_COLOR)
    image1_bgr = cv2.imread(pair["image1_path"], cv2.IMREAD_COLOR)
    if image0_bgr is None or image1_bgr is None:
        raise RuntimeError("Failed to read image pair.")

    image0_gray = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
    image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = image0_gray.shape
    h1, w1 = image1_gray.shape

    image0_rs = cv2.resize(image0_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image1_rs = cv2.resize(image1_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    scale0 = np.array([w0 / img_size, h0 / img_size], dtype=np.float32)
    scale1 = np.array([w1 / img_size, h1 / img_size], dtype=np.float32)
    return image0_bgr, image1_bgr, image0_rs, image1_rs, scale0, scale1


def match_pair(model, pair, args, device):
    image0_bgr, image1_bgr, image0_rs, image1_rs, scale0, scale1 = read_pair_images(pair, args.img_size)

    image0 = torch.from_numpy(image0_rs)[None, None].to(device).float() / 255.0
    image1 = torch.from_numpy(image1_rs)[None, None].to(device).float() / 255.0
    if args.half and device.type == "cuda":
        image0 = image0.half()
        image1 = image1.half()

    batch = {"image0": image0, "image1": image1}

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - start) * 1000.0

    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy().astype(np.float32) * scale0
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy().astype(np.float32) * scale1
    return image0_bgr, image1_bgr, mkpts0, mkpts1, runtime_ms


def transform_points(points, matrix):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    warped_h = points_h @ matrix.T
    denom = warped_h[:, 2:3]
    valid = np.abs(denom[:, 0]) > 1e-8
    warped = np.full((len(points), 2), np.nan, dtype=np.float32)
    warped[valid] = warped_h[valid, :2] / denom[valid]
    return warped


def paper_errors(mkpts0, mkpts1, matrix_0to1):
    """Paper protocol: compare transformed p' with q under the ground-truth label."""
    if len(mkpts0) == 0:
        return np.empty((0,), dtype=np.float32)
    warped0 = transform_points(mkpts0, matrix_0to1)
    errors = np.linalg.norm(warped0 - mkpts1, axis=1).astype(np.float32)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def compute_paper_metrics(pair, mkpts0, mkpts1, errors, args, runtime_ms):
    correct_mask = errors <= args.correct_thr
    ncm = int(correct_mask.sum())
    num_matches = int(len(mkpts0))
    pre = ncm / num_matches if num_matches > 0 else 0.0
    success = ncm >= args.success_ncm

    if success:
        correct_errors = errors[correct_mask]
        rmse = float(np.sqrt(np.mean(correct_errors**2))) if len(correct_errors) else args.failed_rmse
    else:
        rmse = args.failed_rmse

    return {
        "scene": pair["scene"],
        "scene_pair": pair["scene_pair"],
        "modality_pair": pair["modality_pair"],
        "folder": pair["folder"],
        "id": pair.get("id", ""),
        "image0": Path(pair["image0_path"]).name,
        "image1": Path(pair["image1_path"]).name,
        "matrix": Path(pair["matrix_path"]).name,
        "matches": num_matches,
        "NCM": ncm,
        "Pre": pre,
        "SR": 1 if success else 0,
        "RMSE": rmse,
        "runtime_ms": runtime_ms,
    }


def summarize_rows(rows):
    if not rows:
        return {
            "num_pairs": 0,
            "NCM": 0.0,
            "Pre": 0.0,
            "SR": 0.0,
            "RMSE": 0.0,
            "mean_matches": 0.0,
            "mean_runtime_ms": 0.0,
        }

    finite_runtime = [row["runtime_ms"] for row in rows if np.isfinite(row["runtime_ms"])]
    return {
        "num_pairs": len(rows),
        "NCM": float(np.mean([row["NCM"] for row in rows])),
        "Pre": float(np.mean([row["Pre"] for row in rows])),
        "SR": float(np.mean([row["SR"] for row in rows])),
        "RMSE": float(np.mean([row["RMSE"] for row in rows])),
        "mean_matches": float(np.mean([row["matches"] for row in rows])),
        "mean_runtime_ms": float(np.mean(finite_runtime)) if finite_runtime else 0.0,
    }


def summarize(rows, args):
    groups = defaultdict(list)
    for row in rows:
        groups[row["modality_pair"]].append(row)

    return {
        "protocol": {
            "NCM": f"number of matches with ||T(p)-q||_2 <= {args.correct_thr} pixels",
            "Pre": "NCM divided by total produced matches for each image pair",
            "SR": f"success ratio of pairs with NCM >= {args.success_ncm}",
            "RMSE": (
                "sqrt(mean squared reprojection error of correct matches) for successful pairs; "
                f"{args.failed_rmse} for failed pairs"
            ),
            "filtering": "ground-truth label only; no RANSAC outlier removal",
            "SwinMatcher_thresholds": {"theta_c": args.coarse_thr, "theta_f": args.fine_thr},
        },
        "overall": summarize_rows(rows),
        "by_modality_pair": {name: summarize_rows(group_rows) for name, group_rows in sorted(groups.items())},
    }


def safe_name(name):
    keep = []
    for char in name:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def draw_matches(image0, image1, mkpts0, mkpts1, correct_mask):
    h0, h1 = image0.shape[0], image1.shape[0]
    if h0 != h1:
        if h0 < h1:
            pad_top = (h1 - h0) // 2
            pad_bottom = h1 - h0 - pad_top
            image0 = cv2.copyMakeBorder(image0, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT)
            mkpts0 = mkpts0.copy()
            mkpts0[:, 1] += pad_top
        else:
            pad_top = (h0 - h1) // 2
            pad_bottom = h0 - h1 - pad_top
            image1 = cv2.copyMakeBorder(image1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT)
            mkpts1 = mkpts1.copy()
            mkpts1[:, 1] += pad_top

    vis = np.concatenate([image0, image1], axis=1)
    x_offset = image0.shape[1]
    for p0, p1, correct in zip(mkpts0, mkpts1, correct_mask):
        color = (0, 220, 0) if correct else (0, 0, 255)
        p0_i = tuple(np.round(p0).astype(int))
        p1_i = tuple(np.round([p1[0] + x_offset, p1[1]]).astype(int))
        cv2.circle(vis, p0_i, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, p1_i, 2, color, -1, lineType=cv2.LINE_AA)
        cv2.line(vis, p0_i, p1_i, color, 1, lineType=cv2.LINE_AA)
    return vis


def write_outputs(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "pair_metrics.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path = output_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    modality_csv_path = output_dir / "modality_metrics.csv"
    modality_fields = [
        "modality_pair",
        "num_pairs",
        "NCM",
        "Pre",
        "SR",
        "RMSE",
        "mean_matches",
        "mean_runtime_ms",
    ]
    with modality_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=modality_fields)
        writer.writeheader()
        for name, metrics in sorted(summary["by_modality_pair"].items()):
            writer.writerow({"modality_pair": name, **metrics})

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["modality_pair"]].append(row)

    modality_dir = output_dir / "by_modality_pair"
    modality_dir.mkdir(parents=True, exist_ok=True)
    for name, group_rows in sorted(grouped_rows.items()):
        group_dir = modality_dir / safe_name(name)
        group_dir.mkdir(parents=True, exist_ok=True)

        group_csv_path = group_dir / "pair_metrics.csv"
        if group_rows:
            with group_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
                writer.writeheader()
                writer.writerows(group_rows)

        group_summary = {
            "modality_pair": name,
            "protocol": summary["protocol"],
            "overall": summarize_rows(group_rows),
        }
        with (group_dir / "summary.json").open("w") as f:
            json.dump(group_summary, f, indent=2, ensure_ascii=False)

    return csv_path, json_path, modality_csv_path, modality_dir


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.half and device.type != "cuda":
        raise ValueError("--half requires CUDA.")

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_path:
        pairs, skipped_without_matrix = load_manifest_pairs(args.manifest_path)
        data_label = args.manifest_path
    else:
        pairs, skipped_without_matrix = build_pairs(args.data_root)
        data_label = args.data_root
    pairs = prepare_pairs(pairs, args)
    if not pairs:
        raise RuntimeError(f"No evaluation pairs found from {data_label}")

    print(f"data: {data_label}")
    print(f"pairs: {len(pairs)}")
    if skipped_without_matrix:
        print(f"skipped_without_matrix: {skipped_without_matrix}")
    print(f"device: {device}")
    print(
        "paper protocol: "
        f"correct if reprojection error <= {args.correct_thr}px; "
        f"success if NCM >= {args.success_ncm}; failed RMSE = {args.failed_rmse}; no RANSAC"
    )

    model = load_model(args, device)
    rows = []

    for index, pair in enumerate(tqdm(pairs, desc="Evaluating")):
        try:
            matrix_0to1 = load_matrix(pair["matrix_path"])
            if pair.get("gt_direction") == "1to0":
                matrix_0to1 = np.linalg.inv(matrix_0to1).astype(np.float32)
            image0_bgr, image1_bgr, mkpts0, mkpts1, runtime_ms = match_pair(model, pair, args, device)
            errors = paper_errors(mkpts0, mkpts1, matrix_0to1)
            row = compute_paper_metrics(pair, mkpts0, mkpts1, errors, args, runtime_ms)
            rows.append(row)

            if args.save_vis and index < args.vis_limit:
                correct_mask = errors <= args.correct_thr
                vis = draw_matches(image0_bgr, image1_bgr, mkpts0, mkpts1, correct_mask)
                out_name = (
                    f"{index:05d}_{pair['scene_pair']}_{pair['folder']}_"
                    f"{Path(pair['image1_path']).stem}_NCM{row['NCM']}_Pre{row['Pre']:.3f}.jpg"
                )
                cv2.imwrite(str(vis_dir / out_name), vis)
        except Exception as exc:
            rows.append(
                {
                    "scene": pair["scene"],
                    "scene_pair": pair["scene_pair"],
                    "modality_pair": pair["modality_pair"],
                    "folder": pair["folder"],
                    "id": pair.get("id", ""),
                    "image0": Path(pair["image0_path"]).name,
                    "image1": Path(pair["image1_path"]).name,
                    "matrix": Path(pair["matrix_path"]).name,
                    "matches": 0,
                    "NCM": 0,
                    "Pre": 0.0,
                    "SR": 0,
                    "RMSE": args.failed_rmse,
                    "runtime_ms": np.nan,
                    "error": str(exc),
                }
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(rows, args)
    csv_path, json_path, modality_csv_path, modality_dir = write_outputs(rows, summary, output_dir)

    print("\nOverall")
    for key, value in summary["overall"].items():
        print(f"{key}: {value}")
    print("\nBy modality pair")
    for name, metrics in summary["by_modality_pair"].items():
        print(f"{name}: NCM={metrics['NCM']:.3f}, Pre={metrics['Pre']:.4f}, SR={metrics['SR']:.4f}, RMSE={metrics['RMSE']:.4f}")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {modality_csv_path}")
    print(f"Wrote modality folders: {modality_dir}")


if __name__ == "__main__":
    main()
