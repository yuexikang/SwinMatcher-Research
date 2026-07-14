# SwinMatcherBIT Manifest Layout

This project keeps the original dataset files in place under:

- `/home/disk1/Data/datasets/SwinMatcher/BIT_multi-modal_datasets_homography_30_rotation`
- `/home/disk1/Data/datasets/SwinMatcher/BIT_multi-modal_datasets_homography_180_rotation`
- `/home/disk1/Data/datasets/SwinMatcher/SwinMatcher_test_datasets`

Manifests store absolute image and GT paths. The source images and labels are not copied, moved, or renamed.

## Recommended Usage

- Train with `manifests/train_SwinMatcherBIT_gt.jsonl`.
- Evaluate the paper test set with `manifests/test_SwinMatcherBIT_gt.jsonl`.
- Use `_multimodal.jsonl` or `_optical_optical.jsonl` when a controlled subset is needed.

Each JSONL row has:

- `mode: "gt_pairs"`
- `image0`, `image1`
- `gt`, with `gt_format` as `txt` or `npy`
- `gt_direction: "0to1"`
- `split`, `variant`, `scene`, `subset`
- `modality0`, `modality1`, `pair_type`

Counts:

- `train_SwinMatcherBIT_gt.jsonl`: 22484 pairs from h30+h180.
- `test_SwinMatcherBIT_gt.jsonl`: 1000 paper-test pairs.
- `SwinMatcherBIT_gt.jsonl`: 23484 pairs, train plus paper-test.

Training entry:

```bash
scripts/train_swinmatcherbit_manifest.sh
```

Paper-protocol evaluation entry:

```bash
scripts/evaluate_swinmatcherbit_gpu2.sh
```
