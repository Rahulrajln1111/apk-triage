from pathlib import Path

from tools.binary import *
from tools.report import *
from tools.zip_audit import *
from tools.features import *

BASE_DIR = Path(__file__).parent

APK_PATH = BASE_DIR / "apks" / "sample.apk"

# ---------- Binary Verification ----------

raw = read_bytes(APK_PATH)
inspect_zip(APK_PATH)

entries = parse_zip(APK_PATH)
features = extract_features(entries)
report = analyze_features(features)

print("=" * 70)
print("BINARY VERIFICATION")
print("=" * 70)

print(f"File Size : {len(raw):,} bytes")
print(f"ZIP Magic : {has_zip_signature(raw)}")

offset = find_eocd(raw)
print("EOCD Offset:", offset)

fields = read_eocd(raw)
print("Signature      :", fields["signature"])
print("Total Entries  :", fields["total_entries"])
print("CD Size        :", fields["cd_size"])
print("CD Offset      :", fields["cd_offset"])
print("Comment Length :", fields["comment_len"])
print("ZIP64 Sentinel :", fields["is_zip64"])

print()

# ---------- Metadata Audit ----------
print_report(report)

print("=" * 70)
print("ZIP FEATURES")
print("=" * 70)
print(features)