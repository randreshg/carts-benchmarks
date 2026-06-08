from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    parse_checksum,
    parse_e2e_timings,
    parse_kernel_timings,
)


# Representative single-node stream stdout. stream prints per-operation min-time
# diagnostics inside the kernel-timed region; they must carry a non-`kernel.`
# prefix so the runner does not sum them into the one kernel.stream wall-clock.
STREAM_OUTPUT = """\
startup.stream: 12.345678s
stream-phase.copy: 4.200000s
stream-phase.scale: 4.100000s
stream-phase.add: 4.500000s
stream-phase.triad: 4.600000s
kernel.stream: 174.718593s
checksum: 6.607875269531e+15
verification.stream: 0.500000s
cleanup.stream: 0.100000s
e2e.stream: 188.361787s
"""


class StreamKernelTimingTest(unittest.TestCase):
    def test_stream_reports_single_kernel_timing(self) -> None:
        timings = parse_kernel_timings(STREAM_OUTPUT)
        # Only the whole-kernel wall-clock is a kernel timing.
        self.assertEqual(timings, {"stream": 174.718593})

    def test_per_operation_diagnostics_not_counted_as_kernel(self) -> None:
        timings = parse_kernel_timings(STREAM_OUTPUT)
        for op in ("copy", "scale", "add", "triad"):
            self.assertNotIn(op, timings)
        # The runner's kernel time is sum(kernel_timings.values()); with one
        # entry it equals kernel.stream and cannot be inflated by the diagnostics.
        self.assertEqual(sum(timings.values()), 174.718593)

    def test_e2e_and_checksum_still_parse(self) -> None:
        self.assertEqual(parse_e2e_timings(STREAM_OUTPUT), {"stream": 188.361787})
        self.assertEqual(parse_checksum(STREAM_OUTPUT), "6.607875269531e+15")

    def test_legacy_kernel_prefix_would_double_count(self) -> None:
        # Guards the rationale: had the diagnostics kept the `kernel.` prefix,
        # the summed kernel time would exceed the real kernel.stream value.
        legacy = STREAM_OUTPUT.replace("stream-phase.", "kernel.")
        legacy_timings = parse_kernel_timings(legacy)
        self.assertGreater(sum(legacy_timings.values()), 174.718593)


if __name__ == "__main__":
    unittest.main()
