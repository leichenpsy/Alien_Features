import random
import shutil
from pathlib import Path
from collections import Counter

# ---------- CONFIG ----------
original_lists = [
    [11, 12, 18, 17],
    [23, 28, 34, 58],
    [26, 46, 67, 70],
    [121, 194, 175, 187],
]

# CHANGE THIS
source_images_folder = Path("./vectorized")

# CHANGE THIS (must be writable)
output_base_folder = Path("./selected/Testing")
# ----------------------------


def validate_input(data):
    if len(data) != 4:
        raise ValueError(f"Expected 4 lists, got {len(data)}.")
    for i, row in enumerate(data, start=1):
        if len(row) != 4:
            raise ValueError(f"List #{i} must contain exactly 4 numbers, got {len(row)}.")

    flat = [x for row in data for x in row]
    counts = Counter(flat)
    dupes = sorted([n for n, c in counts.items() if c > 1])
    if dupes:
        raise ValueError(f"Each number must appear once across all input lists. Duplicates: {dupes}")


def generate_new_lists(data):
    new_lists = [[] for _ in range(4)]
    for row in data:
        targets = random.sample(range(4), 4)  # permutation: 0,1,2,3
        for value, target_idx in zip(row, targets):
            new_lists[target_idx].append(value)
    return new_lists


def ensure_writable_dir(path: Path) -> Path:
    """Return a writable directory; fallback to /tmp/output if needed."""
    path = path.resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        return path
    except Exception as e:
        print(f"[WARN] Cannot write to {path}: {e}")
        fallback = Path("/tmp/output").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Using fallback output directory: {fallback}")
        return fallback


def find_alien_image(folder: Path, number):
    matches = list(folder.glob(f"alien {number}.*"))
    return matches[0] if matches else None


def create_list_folders(base: Path):
    """Always create list_1..list_4 folders and return them."""
    dest_folders = []
    for i in range(1, 5):
        d = base / f"list_{i}"
        d.mkdir(parents=True, exist_ok=True)
        dest_folders.append(d)
    return dest_folders


def copy_images_to_folders(new_lists, source_folder: Path, output_folder: Path):
    dest_folders = create_list_folders(output_folder)
    missing = []
    copied = 0

    for i, lst in enumerate(new_lists):
        dest = dest_folders[i]
        for number in lst:
            img = find_alien_image(source_folder, number)
            if img is None:
                missing.append(number)
                continue
            shutil.copy2(img, dest / img.name)
            copied += 1

    return missing, copied, dest_folders


def main():
    validate_input(original_lists)

    # random.seed(42)  # uncomment for reproducible result
    new_lists = generate_new_lists(original_lists)

    # Resolve/check paths
    src = source_images_folder.resolve()
    out = ensure_writable_dir(output_base_folder)

    print(f"[INFO] Source folder : {src}")
    print(f"[INFO] Output folder : {out}")

    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source folder not found or not a directory: {src}")

    # Print generated lists
    print("\nGenerated new lists:")
    for i, lst in enumerate(new_lists, start=1):
        print(f"list_{i}: {lst}")

    # Create folders + copy
    missing, copied, dest_folders = copy_images_to_folders(new_lists, src, out)

    print("\n[INFO] Created/confirmed destination folders:")
    for d in dest_folders:
        print(" -", d)

    print(f"\n[INFO] Copied files: {copied}")
    if missing:
        print("[WARN] Missing images for numbers:", sorted(set(missing)))
    else:
        print("[INFO] All images found and copied.")


if __name__ == "__main__":
    main()