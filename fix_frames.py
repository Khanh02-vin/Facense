import os
import json
import unicodedata

frames_dir = "d:/Project/Face_project/data/annotations/frames/"

with open("d:/Project/Face_project/data/annotations/annotation_pairs.json", "r", encoding="utf-8") as f:
    pairs_data = json.load(f)

names_in_pairs = {p["identity_A"] for p in pairs_data["pairs"]}
names_in_pairs |= {p["identity_B"] for p in pairs_data["pairs"]}

def name_only(fname):
    return fname.rsplit("_", 1)[0] if "_" in fname else fname

def normalize_name(s):
    """Normalize to NFC, strip diacritics, lowercase → canonical ASCII key."""
    # NFC normalizes NFD (macOS) to NFC (standard)
    nfc = unicodedata.normalize("NFC", s)
    # Strip diacritics: keep only letters and spaces
    stripped = "".join(
        c for c in nfc
        if c.isalpha() or c == " "
    )
    return stripped.lower().strip()

# Build lookup: normalized_key -> original name
norm_to_name = {}
for name in names_in_pairs:
    key = normalize_name(name)
    norm_to_name[key] = name

# Map codepoints (from name_analysis.json) to the expected names:
# These are the 18 files that need renaming:
# The codepoint analysis showed they don't match by direct string comparison.
# Let's manually map by checking what they SHOULD be based on pairs.

# For each name in pairs, normalize it and check if any file
# on disk (when normalized) matches. If a file's normalized form
# matches a pair name, rename it.

files = sorted(os.listdir(frames_dir))

rename_ops = []
skipped = []
already_correct = []

for fname in files:
    if not fname.endswith(".jpg"):
        continue
    base = name_only(fname)
    suffix = fname[len(base):]  # _0.jpg

    if base in names_in_pairs:
        already_correct.append(fname)
        continue

    # Normalize this disk name to canonical form
    norm_key = normalize_name(base)

    if norm_key in norm_to_name:
        correct = norm_to_name[norm_key]
        new_name = correct + suffix
        if base != correct:
            rename_ops.append({"old": fname, "new": new_name, "correct": correct})
        else:
            already_correct.append(fname)
    else:
        skipped.append(fname)

print(f"Already correct: {len(already_correct)}")
print(f"Rename ops: {len(rename_ops)}")
print(f"Skipped: {len(skipped)}")

# Apply renames
for op in rename_ops:
    old_path = os.path.join(frames_dir, op["old"])
    new_path = os.path.join(frames_dir, op["new"])
    os.rename(old_path, new_path)

# Verify
files_after = sorted(os.listdir(frames_dir))
still_bad = [f for f in files_after if f.endswith(".jpg") and name_only(f) not in names_in_pairs]

report = {
    "already_correct_count": len(already_correct),
    "renamed_count": len(rename_ops),
    "renamed": rename_ops,
    "skipped": skipped,
    "still_bad_count": len(still_bad),
    "still_bad": still_bad,
}

with open("d:/Project/Face_project/fix_report.json", "w", encoding="utf-8") as out:
    json.dump(report, out, ensure_ascii=False, indent=2)

print(f"\nStill bad after fix: {len(still_bad)}")
if still_bad:
    for f in still_bad:
        print(f"  {f!r}")
print("See fix_report.json")
