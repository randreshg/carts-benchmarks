"""SLURM integration for CARTS benchmarks.

- batch: Job submission and monitoring orchestration.
- experiment: High-level benchmark build/script/submit/report orchestration.
- job_result: Per-job result parsing (standalone script executed inside SLURM jobs).
- models: Shared dataclasses/constants for SLURM execution.
- results: Aggregated result collection and failure-row synthesis.
"""
