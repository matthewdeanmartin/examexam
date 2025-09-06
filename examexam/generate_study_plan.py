from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter

import dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from examexam.generate_topic_research import generate_study_guide

# Load environment variables
dotenv.load_dotenv()

# ---- Logging setup ----
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_level=True)],
    )

console = Console()


def generate_study_plan_now(toc_file: str, model: str = "openai") -> None:
    """
    Generates a consolidated study guide for all topics in a TOC file.
    """
    toc_path = Path(toc_file)
    if not toc_path.exists():
        console.print(Panel.fit(f"[red]TOC file not found:[/] {toc_file}", title="Error"))
        return

    with toc_path.open(encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    total_topics = len(topics)
    if total_topics == 0:
        console.print(Panel.fit("[yellow]TOC file is empty.[/]", title="Nothing to do"))
        return

    console.rule("[bold]Study Plan Generation[/bold]")
    console.print(f"Generating study guides for [bold]{total_topics}[/] topics using model [italic]{model}[/]…\n")

    all_guides_content = [f"# Study Plan for {toc_path.stem}\n\n"]
    failures = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        expand=True,
        console=console,
        transient=False,
    )

    with progress:
        overall_task = progress.add_task("Overall", total=total_topics)

        for idx, topic in enumerate(topics, start=1):
            topic_task_desc = f"{idx}/{total_topics} {topic[:40]}{'...' if len(topic) > 40 else ''}"
            topic_task = progress.add_task(topic_task_desc, total=1)

            t0 = perf_counter()
            guide_content = generate_study_guide(topic, model)
            dt = perf_counter() - t0

            if guide_content:
                all_guides_content.append(f"## Topic: {topic}\n\n{guide_content}\n\n---\n\n")
                progress.update(
                    topic_task,
                    description=f"{topic_task_desc} [green](ok in {dt:.2f}s)[/]",
                )
            else:
                failures += 1
                progress.update(topic_task, description=f"{topic_task_desc} [red](failed)[/]")

            progress.advance(topic_task)
            progress.advance(overall_task)

    console.rule()

    if not all_guides_content or len(all_guides_content) == 1:
        console.print(Panel.fit("[bold red]Failed to generate any study guides.[/]", title="Complete Failure"))
        return

    # Save the consolidated file
    output_dir = Path("study_guide")
    output_dir.mkdir(exist_ok=True)
    output_filename = f"{toc_path.stem}_study_plan.md"
    output_path = output_dir / output_filename

    try:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("".join(all_guides_content))

        summary_message = (
            f"Successfully generated study guides for [bold green]{total_topics - failures}[/] topics. "
            f"Saved to:\n[bold cyan]{output_path}[/]"
        )
        if failures > 0:
            summary_message += f"\n[bold red]{failures}[/] topics failed."

        console.print(Panel.fit(summary_message, title="Summary"))

    except OSError as e:
        console.print(Panel(f"[bold red]Error saving file: {e}[/]", title="Save Error"))


if __name__ == "__main__":
    # Example of how to run this directly
    # You would need an 'example_toc.txt' file for this to work
    generate_study_plan_now(toc_file="example_toc.txt", model="openai")
