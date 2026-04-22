"""
Generate synthetic MVTec-like data for end-to-end pipeline testing.
Creates realistic directory structure with random images and masks.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_mvtec(root="./data/mvtec", category="carpet",
                           n_train=30, n_test_good=10, n_test_defect=10,
                           img_size=256):
    """Create synthetic MVTec directory structure with images."""
    
    print(f"Creating synthetic '{category}' data at {root}/{category}...")
    
    # Train/good
    train_dir = os.path.join(root, category, "train", "good")
    os.makedirs(train_dir, exist_ok=True)
    for i in range(n_train):
        img = _make_normal_image(img_size)
        img.save(os.path.join(train_dir, f"{i:03d}.png"))
    print(f"  ✅ Train: {n_train} normal images")

    # Test/good
    test_good = os.path.join(root, category, "test", "good")
    os.makedirs(test_good, exist_ok=True)
    for i in range(n_test_good):
        img = _make_normal_image(img_size)
        img.save(os.path.join(test_good, f"{i:03d}.png"))
    print(f"  ✅ Test/good: {n_test_good} images")

    # Test/scratch (defect type)
    test_defect = os.path.join(root, category, "test", "scratch")
    gt_dir = os.path.join(root, category, "ground_truth", "scratch")
    os.makedirs(test_defect, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    for i in range(n_test_defect):
        img, mask = _make_defect_image(img_size)
        img.save(os.path.join(test_defect, f"{i:03d}.png"))
        mask.save(os.path.join(gt_dir, f"{i:03d}_mask.png"))
    print(f"  ✅ Test/scratch: {n_test_defect} images + masks")

    # Test/hole (second defect type)
    test_hole = os.path.join(root, category, "test", "hole")
    gt_hole = os.path.join(root, category, "ground_truth", "hole")
    os.makedirs(test_hole, exist_ok=True)
    os.makedirs(gt_hole, exist_ok=True)
    for i in range(n_test_defect // 2):
        img, mask = _make_defect_image(img_size, defect_type="circle")
        img.save(os.path.join(test_hole, f"{i:03d}.png"))
        mask.save(os.path.join(gt_hole, f"{i:03d}_mask.png"))
    print(f"  ✅ Test/hole: {n_test_defect // 2} images + masks")


def _make_normal_image(size=256):
    """Generate a synthetic 'normal' texture image."""
    # Create a base texture with consistent color/pattern
    np.random.seed(None)
    base_color = np.random.randint(100, 200, 3)
    noise = np.random.normal(0, 10, (size, size, 3))
    img_arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


def _make_defect_image(size=256, defect_type="scratch"):
    """Generate a synthetic image with a defect and its ground truth mask."""
    # Normal background
    img = _make_normal_image(size)
    mask = Image.new("L", (size, size), 0)

    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    if defect_type == "scratch":
        # Draw a scratch line
        x1 = np.random.randint(size // 4, size // 2)
        y1 = np.random.randint(size // 4, size // 2)
        x2 = x1 + np.random.randint(40, 120)
        y2 = y1 + np.random.randint(40, 120)
        width = np.random.randint(3, 8)
        draw_img.line([(x1, y1), (x2, y2)], fill=(30, 30, 30), width=width)
        draw_mask.line([(x1, y1), (x2, y2)], fill=255, width=width + 4)
    else:
        # Draw a circular hole
        cx = np.random.randint(size // 3, 2 * size // 3)
        cy = np.random.randint(size // 3, 2 * size // 3)
        r = np.random.randint(10, 30)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw_img.ellipse(bbox, fill=(20, 20, 20))
        draw_mask.ellipse(bbox, fill=255)

    return img, mask


if __name__ == "__main__":
    create_synthetic_mvtec("./data/mvtec", "carpet", n_train=30, n_test_good=10, n_test_defect=10)
    print("\n✅ Synthetic dataset ready!")
