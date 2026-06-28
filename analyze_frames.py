import os, json, sys

frames_dir = "d:/Project/Face_project/data/annotations/frames/"

# Read raw filenames as bytes from NTFS
def get_raw_filenames(dir_path):
    """Read directory entries as UTF-16LE bytes to get true Unicode names."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    INVALID_HANDLE = -1
    FILE_LIST_DIRECTORY = 1
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    h = kernel32.CreateFileW(dir_path, FILE_LIST_DIRECTORY, 1, None, 3,
                              FILE_FLAG_BACKUP_SEMANTICS, None)
    if h == INVALID_HANDLE:
        return None

    try:
        results = []
        # WIN32_FIND_DATAW structure size is 592 bytes
        buf = ctypes.create_string_buffer(592 * 100)
        n = kernel32.ReadDirectoryFilesW(h, buf, 592 * 100, True, None, None)
        # Parse - each entry is 592 bytes, filename is wchar[] starting at offset 44
        # Just use FindNextFileW in a loop instead
    except Exception:
        pass

    # Fallback: use FindFirstFileW / FindNextFileW
    WIN32_FIND_DATAW = "IIIIIIIIIIIIIiIII16s32s104s520s"
    FindFirstFileW = kernel32.FindFirstFileW
    FindNextFileW = kernel32.FindNextFileW
    FindClose = kernel32.FindClose

    pattern = os.path.join(dir_path, "*")
    fd_buf = ctypes.create_string_buffer(592)

    names = []
    hFind = FindFirstFileW(pattern, fd_buf)
    if hFind != INVALID_HANDLE:
        try:
            while True:
                # ctypes doesn't easily parse WIN32_FIND_DATAW, so let's use ctypes.wintypes
                pass
        except:
            pass
        FindClose(hFind)

    return None

# Simple approach: the file names as Python sees them (Unicode strings)
# The issue: Python 3 on Windows reads NTFS as Unicode (UTF-16LE internally).
# But when printing to terminal (cp1252), cp1252-unencodable chars fail.
# What we see in repr() is the correct Unicode string.

# Let me verify: what are the ACTUAL Unicode codepoints?
disk_files = sorted(os.listdir(frames_dir))

with open("d:/Project/Face_project/data/annotations/annotation_pairs.json", "r", encoding="utf-8") as f:
    pairs_data = json.load(f)

names_in_pairs = set()
for p in pairs_data["pairs"]:
    names_in_pairs.add(p["identity_A"])
    names_in_pairs.add(p["identity_B"])

def name_only(fname):
    return fname.rsplit("_", 1)[0] if "_" in fname else fname

# Write all names to JSON - this will work even if print fails
output = {}
for fname in disk_files:
    if not fname.endswith(".jpg"):
        continue
    base = name_only(fname)
    in_pairs = base in names_in_pairs
    codepoints = [hex(ord(c)) for c in base]
    output[fname] = {
        "in_pairs": in_pairs,
        "codepoints": codepoints,
        "base": base
    }

with open("d:/Project/Face_project/name_analysis.json", "w", encoding="utf-8") as out:
    json.dump(output, out, ensure_ascii=False, indent=2)

# Count matches
matched = sum(1 for v in output.values() if v["in_pairs"])
unmatched = sum(1 for v in output.values() if not v["in_pairs"])
print(f"Matched: {matched}, Unmatched: {unmatched}")
