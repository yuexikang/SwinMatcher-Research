import os
import cv2
import torch
import warnings
import numpy as np
from tqdm import tqdm

from src.swinmatcher import SwinMatcher, default_cfg

warnings.filterwarnings("ignore")


def draw_matches(image0, image1, points0, points1):
    difference = image0.shape[0] - image1.shape[0]

    if difference < 0:
        top = abs(difference) // 2
        bottom = abs(difference) - top
        image0 = cv2.copyMakeBorder(image0, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points0) > 0:
            points0[:, 1] += top
    elif difference > 0:
        top = difference // 2
        bottom = difference - top
        image1 = cv2.copyMakeBorder(image1, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        if len(points1) > 0:
            points1[:, 1] += top

    kpts0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in points0]
    kpts1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in points1]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(kpts0))]

    return cv2.drawMatches(image0, kpts0, image1, kpts1, matches, None, flags=2)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == "__main__":
    ckpt_path = "weights/swinmatcher_512.ckpt"
    folder_path = "sample"
    save_dir = "outputs_demo"

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Cannot find sample folder: {folder_path}")

    image_names = sorted([
        x for x in os.listdir(folder_path)
        if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ])

    if len(image_names) < 2:
        raise RuntimeError("sample folder has fewer than 2 images.")

    if len(image_names) % 2 != 0:
        print(f"[Warning] Odd number of images: {len(image_names)}. The last one will be ignored.")

    config = default_cfg
    config["match_coarse"]["thr"] = 0.3
    config["match_coarse"]["border_rm"] = 2
    config["fine"]["thr"] = 0.1
    config["img_size"] = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half_precision = device == "cuda"

    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    matcher = SwinMatcher(config=config)

    ckpt = load_checkpoint(ckpt_path, device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    matcher.load_state_dict(state_dict)

    matcher = matcher.eval().to(device)

    if half_precision:
        matcher = matcher.half()

    for i in tqdm(range(0, len(image_names) - 1, 2)):
        name0 = image_names[i]
        name1 = image_names[i + 1]

        path0 = os.path.join(folder_path, name0)
        path1 = os.path.join(folder_path, name1)

        image0_bgr = cv2.imread(path0)
        image1_bgr = cv2.imread(path1)

        if image0_bgr is None or image1_bgr is None:
            print(f"[Skip] Failed to read: {path0}, {path1}")
            continue

        image0_gray = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
        image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)

        h0, w0 = image0_gray.shape
        h1, w1 = image1_gray.shape

        img_size = config["img_size"]

        image0_gray_rs = cv2.resize(image0_gray, (img_size, img_size))
        image1_gray_rs = cv2.resize(image1_gray, (img_size, img_size))

        scale0 = np.array([w0 / img_size, h0 / img_size], dtype=np.float32)
        scale1 = np.array([w1 / img_size, h1 / img_size], dtype=np.float32)

        image0 = torch.from_numpy(image0_gray_rs)[None, None].to(device).float() / 255.0
        image1 = torch.from_numpy(image1_gray_rs)[None, None].to(device).float() / 255.0

        if half_precision:
            image0 = image0.half()
            image1 = image1.half()

        batch = {
            "image0": image0,
            "image1": image1,
        }

        with torch.no_grad():
            matcher(batch)
            matched_points0 = batch["mkpts0_f"].detach().cpu().numpy() * scale0
            matched_points1 = batch["mkpts1_f"].detach().cpu().numpy() * scale1

        raw_matches = len(matched_points0)

        if len(matched_points0) >= 4:
            H, mask = cv2.findHomography(
                matched_points1,
                matched_points0,
                cv2.USAC_MAGSAC,
                5.0
            )
            if mask is not None:
                mask = mask.flatten().astype(bool)
                matched_points0 = matched_points0[mask]
                matched_points1 = matched_points1[mask]

        inlier_matches = len(matched_points0)

        vis = draw_matches(image0_bgr, image1_bgr, matched_points0.copy(), matched_points1.copy())

        out_name = f"pair_{i//2:03d}_{raw_matches}_raw_{inlier_matches}_inlier.jpg"
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, vis)

        print(f"{name0} <-> {name1}: raw={raw_matches}, inlier={inlier_matches}, saved={out_path}")

    print(f"Done. Results saved in: {save_dir}")