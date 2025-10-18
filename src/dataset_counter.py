import os
from collections import defaultdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def get_folder_size(path: str) -> int:
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return total_size

def CountDataset(dataset_path: str) -> dict:
    class_counts = defaultdict(int)
    folder_sizes = {}

    # ✅ Get all valid subfolders
    try:
        fruit_folders = sorted([
            f for f in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')
        ])
    except FileNotFoundError:
        print(f"❌ Folder not found: {dataset_path}")
        return {}
    except PermissionError:
        print(f"⚠️ Permission denied: {dataset_path}")
        return {}

    # ✅ Count images & size per class
    for fruit_folder in fruit_folders:
        fruit_path = os.path.join(dataset_path, fruit_folder)
        count = 0
        for root, _, files in os.walk(fruit_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    count += 1
        class_counts[fruit_folder] = count
        folder_sizes[fruit_folder] = get_folder_size(fruit_path)

    if not class_counts:
        print("⚠️ No images found in dataset.")
        return {}

    # ✅ Compute summary values
    max_class_count = max(class_counts.values())
    total_images = sum(class_counts.values())
    total_size_bytes = sum(folder_sizes.values())

    # ✅ Build dataset_info dictionary
    dataset_info = {}
    for fruit, count in class_counts.items():
        ratio = count / max_class_count if count > 0 else 0
        needed = max_class_count - count
        size_mb = folder_sizes[fruit] / (1024 ** 2)
        dataset_info[fruit] = {
            "count": count,
            "ratio": ratio,
            "needed": needed,
            "size_mb": size_mb
        }

    dataset_info["total_images"] = total_images
    dataset_info["max_class_count"] = max_class_count
    dataset_info["total_size_mb"] = total_size_bytes / (1024 ** 2)

    # ✅ Dynamic column widths for pretty printing
    class_w = max(10, max((len(f) for f in fruit_folders), default=10) + 2)
    count_w = max(7, max((len(str(dataset_info[f]["count"])) for f in fruit_folders), default=7) + 2)
    ratio_w = 12
    pad_between_ratio_needed = 3
    needed_w = max(8, max((len(str(dataset_info[f]["needed"])) for f in fruit_folders), default=8) + 3)
    size_w = 12
    mb_suffix = " MB"

    # ✅ Compute separator length for clean borders
    sep_len = class_w + count_w + ratio_w + pad_between_ratio_needed + needed_w + size_w + len(mb_suffix) + 8

    # ✅ Print Header
    print("\n📊 DATASET SUMMARY")
    print("=" * sep_len)
    header = (
        f"{'Class':<{class_w}}"
        f"{'count':>{count_w}}"
        f"{'ratio':>{ratio_w}}"
        f"{' ' * pad_between_ratio_needed}"
        f"{'needed':>{needed_w}}"
        f"{'size_mb':>{size_w + len(mb_suffix)}}"
    )
    print(header)
    print("=" * sep_len)

    # ✅ Print each fruit
    for fruit in fruit_folders:
        info = dataset_info[fruit]
        print(
            f"{fruit:<{class_w}}"
            f"{info['count']:>{count_w}d}"
            f"{info['ratio']:>{ratio_w}.4f}"
            f"{' ' * pad_between_ratio_needed}"
            f"{info['needed']:>{needed_w}d}"
            f"{info['size_mb']:>{size_w}.2f}{mb_suffix}"
        )

    # ✅ Print summary values
    print("=" * sep_len)
    print(f"{'total_images':<{class_w}}{dataset_info['total_images']:>{count_w}}")
    print(f"{'max_class_count':<{class_w}}{dataset_info['max_class_count']:>{count_w}}")
    print(
        f"{'total_size_mb':<{class_w}}"
        f"{dataset_info['total_size_mb']:>{count_w + ratio_w + pad_between_ratio_needed + needed_w + size_w}.2f}{mb_suffix}"
    )
    print("=" * sep_len)

    return dataset_info
