"""Rich formatting helpers for benchmark CLI output."""

from __future__ import annotations

from dekk import Colors, Symbols, console


def format_passed(count: int) -> str:
    return f"[{Colors.SUCCESS}]{Symbols.PASS} {count}[/{Colors.SUCCESS}] passed"


def format_failed(count: int) -> str:
    return f"[{Colors.ERROR}]{Symbols.FAIL} {count}[/{Colors.ERROR}] failed"


def format_skipped(count: int) -> str:
    return f"[{Colors.DEBUG}]{Symbols.SKIP} {count}[/{Colors.DEBUG}] skipped"


def format_summary_line(passed: int, failed: int, skipped: int) -> str:
    return f"{format_passed(passed)}  {format_failed(failed)}  {format_skipped(skipped)}"


def print_footer(title: str, style: str = Colors.SUCCESS) -> None:
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    console.print()
    console.print(
        Panel(
            Text.from_markup(title, style=style),
            box=box.HEAVY,
            border_style=style,
            padding=(0, 1),
        )
    )
    console.print()
