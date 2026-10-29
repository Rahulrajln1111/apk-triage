from dataclasses import dataclass
from tools.features import ZipFeatures


@dataclass
class Finding:
    title: str
    severity: str      # LOW / MEDIUM / HIGH
    value: str


@dataclass
class ZipReport:
    score: int
    findings: list[Finding]


def print_report(report: ZipReport):
    print("\n" + "=" * 60)
    print("FORENSIC SECURITY REPORT")
    print("=" * 60)

    print(f"Risk Score : {report.score}/100\n")

    for item in report.findings:
        print(f"[{item.severity}] {item.title}: {item.value}")


def analyze_features(f: ZipFeatures) -> ZipReport:
    findings = []
    score = 0

    if f.sequential_dex_score > 20:
        findings.append(Finding("Large DEX family", "HIGH", str(f.sequential_dex_score)))
        score += 30

    if f.max_ratio > 100:
        findings.append(Finding("Extreme compression ratio", "HIGH", f"{f.max_ratio:.1f}x"))
        score += 25

    if f.path_traversal_count > 0:
        findings.append(Finding("Path traversal entries", "HIGH", str(f.path_traversal_count)))
        score += 35

    if f.absolute_path_count > 0:
        findings.append(Finding("Absolute path entries", "MEDIUM", str(f.absolute_path_count)))
        score += 10

    if f.duplicate_name_count > 0:
        findings.append(Finding("Duplicate normalized names", "MEDIUM", str(f.duplicate_name_count)))
        score += 15

    if f.null_byte_name_count > 0:
        findings.append(Finding("Null-byte filenames", "HIGH", str(f.null_byte_name_count)))
        score += 25

    if f.symlink_count > 0:
        findings.append(Finding("Symlink entries", "HIGH", str(f.symlink_count)))
        score += 20

    if f.encrypted_entry_count > 0:
        findings.append(Finding("Encrypted archive entries", "MEDIUM", str(f.encrypted_entry_count)))
        score += 10

    if f.zip64_present > 0:
        findings.append(Finding("ZIP64 extra fields", "MEDIUM", str(f.zip64_present)))
        score += 10

    if f.data_descriptor_count > 0:
        findings.append(Finding("Data descriptor usage", "LOW", str(f.data_descriptor_count)))

    if f.nested_archive_count > 0:
        findings.append(Finding("Nested archive candidates", "HIGH", str(f.nested_archive_count)))
        score += 20

    if f.directory_depth_entropy > 2:
        findings.append(Finding("Irregular directory structure", "MEDIUM", f"{f.directory_depth_entropy:.2f}"))
        score += 10

    # NOTE: traversal/duplicate signals come from the ZipFeatures computed over
    # the audited entries, so an extra audit-derived finding here only printed
    # the same issue twice. Callers hand in the audit's entries instead.
    return ZipReport(min(score, 100), findings)