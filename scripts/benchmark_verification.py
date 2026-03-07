"""Shared checksum verification helpers for local and SLURM benchmark paths."""

from __future__ import annotations

from typing import Optional

from benchmark_models import Status, VerificationMode, VerificationResult


def _compare_checksum_values(
    lhs: str,
    rhs: str,
    tolerance: float,
) -> bool:
    try:
        lhs_val = float(lhs)
        rhs_val = float(rhs)
    except ValueError:
        return lhs.strip() == rhs.strip()

    baseline = max(abs(rhs_val), 1e-10)
    return abs(lhs_val - rhs_val) / baseline <= tolerance


def verify_against_omp(
    arts_status: Status,
    arts_checksum: Optional[str],
    omp_status: Status,
    omp_checksum: Optional[str],
    tolerance: float,
) -> VerificationResult:
    """Verify ARTS output directly against an OpenMP run."""
    if arts_status != Status.PASS or omp_status != Status.PASS:
        return VerificationResult(
            correct=False,
            arts_checksum=arts_checksum,
            omp_checksum=omp_checksum,
            tolerance_used=tolerance,
            note="Cannot verify: one or both runs failed",
            mode=VerificationMode.DIRECT_OMP.value,
        )

    if arts_checksum is None or omp_checksum is None:
        return VerificationResult(
            correct=False,
            arts_checksum=arts_checksum,
            omp_checksum=omp_checksum,
            tolerance_used=tolerance,
            note="Cannot verify: checksum not found in output",
            mode=VerificationMode.DIRECT_OMP.value,
        )

    correct = _compare_checksum_values(arts_checksum, omp_checksum, tolerance)
    if correct:
        note = "Checksums match within tolerance"
    else:
        note = f"Checksum mismatch: ARTS={arts_checksum}, OMP={omp_checksum}"

    return VerificationResult(
        correct=correct,
        arts_checksum=arts_checksum,
        omp_checksum=omp_checksum,
        tolerance_used=tolerance,
        note=note,
        mode=VerificationMode.DIRECT_OMP.value,
    )


def verify_against_reference(
    arts_status: Status,
    arts_checksum: Optional[str],
    reference_checksum: Optional[str],
    tolerance: float,
    *,
    reference_source: Optional[str] = None,
    reference_omp_threads: Optional[int] = None,
) -> VerificationResult:
    """Verify ARTS output against a stored OpenMP reference checksum."""
    if arts_status != Status.PASS:
        return VerificationResult(
            correct=False,
            arts_checksum=arts_checksum,
            omp_checksum=None,
            tolerance_used=tolerance,
            note="Cannot verify: ARTS run failed",
            mode=VerificationMode.STORED_OMP_REFERENCE.value,
            reference_checksum=reference_checksum,
            reference_source=reference_source,
            reference_omp_threads=reference_omp_threads,
        )

    if arts_checksum is None or reference_checksum is None:
        return VerificationResult(
            correct=False,
            arts_checksum=arts_checksum,
            omp_checksum=None,
            tolerance_used=tolerance,
            note="Cannot verify: checksum not found in output",
            mode=VerificationMode.STORED_OMP_REFERENCE.value,
            reference_checksum=reference_checksum,
            reference_source=reference_source,
            reference_omp_threads=reference_omp_threads,
        )

    correct = _compare_checksum_values(arts_checksum, reference_checksum, tolerance)
    if correct:
        note = "Checksum matches stored OMP reference within tolerance"
    else:
        note = (
            f"Checksum mismatch: ARTS={arts_checksum}, "
            f"stored OMP reference={reference_checksum}"
        )

    return VerificationResult(
        correct=correct,
        arts_checksum=arts_checksum,
        omp_checksum=None,
        tolerance_used=tolerance,
        note=note,
        mode=VerificationMode.STORED_OMP_REFERENCE.value,
        reference_checksum=reference_checksum,
        reference_source=reference_source,
        reference_omp_threads=reference_omp_threads,
    )
