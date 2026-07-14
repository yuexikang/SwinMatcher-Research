import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build SwinMatcherBIT gt_pairs manifests.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/disk1/Data/datasets/SwinMatcher"),
        help="Root containing SwinMatcher dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("manifests"),
        help="Where JSONL manifests will be written.",
    )
    return parser.parse_args()


def clean_variant(name):
    if name == "BIT_multi-modal_datasets_homography_30_rotation":
        return "h30"
    if name == "BIT_multi-modal_datasets_homography_180_rotation":
        return "h180"
    return name.lower().replace("-", "_")


def strip_scene_prefix(scene, subset):
    prefix = f"{scene}_"
    return subset[len(prefix):] if subset.startswith(prefix) else subset


def normalize_token(token):
    token = token.lower()
    if token in {"rgb", "rgb1", "rgb2", "rgb3", "opt", "optical"}:
        return "optical"
    if token in {"sar", "sar1", "sar2"}:
        return "sar"
    if token == "lidar":
        return "lidar"
    if token == "depth":
        return "depth"
    if token in {"ir", "infrared"}:
        return "infrared"
    if token in {"map", "map1", "map2"}:
        return "map"
    if token == "day":
        return "optical_day"
    if token == "night":
        return "optical_night"
    return token


def modality_tokens(scene, subset):
    raw = strip_scene_prefix(scene, subset)
    raw = raw.replace(" ", "_").replace("-", "_")
    parts = [p for p in raw.split("_") if p and not p.isdigit()]
    if len(parts) < 2:
        return "unknown", "unknown"
    return normalize_token(parts[0]), normalize_token(parts[1])


def pair_type(modality0, modality1):
    if modality0 == "optical" and modality1 == "optical":
        return "optical_optical"
    return "multimodal"


def sorted_images(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def candidate_images(images):
    if len(images) < 2:
        return []
    return images[2:] if len(images) > 2 else images[1:]


def make_id(*parts):
    return "/".join(str(p).replace(" ", "_") for p in parts if str(p))


def homography_records(variant_root, split):
    variant = clean_variant(variant_root.name)
    records = []
    skipped = 0

    for scene_dir in sorted(p for p in variant_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        for subset_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            subset = subset_dir.name
            modality0, modality1 = modality_tokens(scene, subset)
            ptype = pair_type(modality0, modality1)
            for folder in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
                images = sorted_images(folder)
                if len(images) < 2:
                    continue
                image0 = images[0]
                for image1 in candidate_images(images):
                    gt = folder / f"{image1.stem}.txt"
                    if not gt.exists():
                        skipped += 1
                        continue
                    records.append(
                        {
                            "dataset": "SwinMatcherBIT",
                            "variant": variant,
                            "id": make_id("swinmatcherbit", variant, scene, subset, folder.name, image1.stem),
                            "split": split,
                            "mode": "gt_pairs",
                            "pair_type": ptype,
                            "image0": str(image0),
                            "image1": str(image1),
                            "gt": str(gt),
                            "gt_format": "txt",
                            "gt_direction": "0to1",
                            "modality0": modality0,
                            "modality1": modality1,
                            "scene": scene,
                            "subset": subset,
                        }
                    )
    return records, skipped


def test_records(test_root):
    records = []
    skipped = 0

    for collection_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        collection = collection_dir.name
        for subset_dir in sorted(p for p in collection_dir.iterdir() if p.is_dir()):
            subset = subset_dir.name
            pseudo_scene = subset.split()[0].split("_")[0]
            modality0, modality1 = modality_tokens(pseudo_scene, subset)
            ptype = pair_type(modality0, modality1)
            for folder in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
                images = sorted_images(folder)
                if len(images) < 2:
                    continue
                image0 = images[0]
                for image1 in candidate_images(images):
                    gt = folder / f"{image1.stem}.npy"
                    if not gt.exists():
                        skipped += 1
                        continue
                    records.append(
                        {
                            "dataset": "SwinMatcherBIT",
                            "variant": "paper_test",
                            "id": make_id("swinmatcherbit", "paper_test", collection, subset, folder.name, image1.stem),
                            "split": "test",
                            "mode": "gt_pairs",
                            "pair_type": ptype,
                            "image0": str(image0),
                            "image1": str(image1),
                            "gt": str(gt),
                            "gt_format": "npy",
                            "gt_direction": "0to1",
                            "modality0": modality0,
                            "modality1": modality1,
                            "scene": collection,
                            "subset": subset,
                        }
                    )
    return records, skipped


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records, skipped):
    by_split = Counter(r["split"] for r in records)
    by_pair_type = Counter(r["pair_type"] for r in records)
    by_variant = Counter(r["variant"] for r in records)
    by_subset = Counter(r["subset"] for r in records)
    by_modality = Counter(f"{r['modality0']}-{r['modality1']}" for r in records)
    return {
        "records": len(records),
        "skipped_without_gt": skipped,
        "by_split": dict(sorted(by_split.items())),
        "by_pair_type": dict(sorted(by_pair_type.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "by_subset": dict(sorted(by_subset.items())),
        "by_modality": dict(sorted(by_modality.items())),
    }


def write_manifest_set(output_dir, name, records):
    write_jsonl(output_dir / f"{name}.jsonl", records)
    for ptype in ["multimodal", "optical_optical"]:
        subset_records = [r for r in records if r["pair_type"] == ptype]
        if subset_records:
            write_jsonl(output_dir / f"{name}_{ptype}.jsonl", subset_records)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    total_skipped = 0
    summaries = {}

    variants = [
        ("BIT_multi-modal_datasets_homography_30_rotation", "train"),
        ("BIT_multi-modal_datasets_homography_180_rotation", "train"),
    ]
    for variant_name, split in variants:
        variant_root = args.data_root / variant_name
        if not variant_root.exists():
            continue
        records, skipped = homography_records(variant_root, split)
        variant = clean_variant(variant_name)
        write_manifest_set(args.output_dir, f"train_SwinMatcherBIT_{variant}_gt", records)
        write_manifest_set(args.output_dir, f"SwinMatcherBIT_{variant}_gt", records)
        summaries[f"SwinMatcherBIT_{variant}"] = summarize(records, skipped)
        all_records.extend(records)
        total_skipped += skipped

    test_root = args.data_root / "SwinMatcher_test_datasets"
    if test_root.exists():
        records, skipped = test_records(test_root)
        write_manifest_set(args.output_dir, "test_SwinMatcherBIT_gt", records)
        summaries["SwinMatcherBIT_paper_test"] = summarize(records, skipped)
        all_records.extend(records)
        total_skipped += skipped

    train_records = [record for record in all_records if record["split"] == "train"]
    if train_records:
        write_manifest_set(args.output_dir, "train_SwinMatcherBIT_gt", train_records)
        summaries["SwinMatcherBIT_train"] = summarize(train_records, 0)

    write_jsonl(args.output_dir / "SwinMatcherBIT_gt.jsonl", all_records)
    summaries["SwinMatcherBIT_all"] = summarize(all_records, total_skipped)

    with (args.output_dir / "SwinMatcherBIT_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
