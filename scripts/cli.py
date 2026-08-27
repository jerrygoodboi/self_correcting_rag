import os
import sys
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag_service import rag_service
from app.services.vector_service import vector_service
from app.services.document_service import document_service

console = Console()


def print_banner():
    console.print(
        Panel(
            "[bold cyan]Self-Correcting Agentic RAG System[/bold cyan]\n"
            "[italic green]Powered by LangGraph, ChromaDB, PostgreSQL & Ollama/Gemini[/italic green]\n"
            "[dim]Commands: /seed, /ingest <path>, /stats, /help, exit[/dim]",
            border_style="bright_blue",
        )
    )


def interactive_repl():
    print_banner()

    thread_id = str(uuid.uuid4())[:8]
    console.print(f"[bold yellow]Session Thread ID:[/bold yellow] [bold green]{thread_id}[/bold green]")
    console.print(f"[bold yellow]Indexed Vector Chunks in ChromaDB:[/bold yellow] [bold white]{vector_service().count()}[/bold white]\n")

    while True:
        try:
            query = console.input("[bold cyan]Question ❯ [/bold cyan]").strip()
            if not query:
                continue

            # Command handling
            if query.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if query.startswith("/"):
                parts = query.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/seed":
                    with console.status("[bold green]Seeding sample documents...[/bold green]"):
                        count = document_service.ingest_directory("data/sample_docs")
                    console.print(f"[green]✓ Re-seeded {count} chunks. Total in DB: {vector_service().count()}[/green]\n")
                    continue

                elif cmd == "/ingest":
                    if not arg:
                        console.print("[red]Usage: /ingest <path/to/file>[/red]\n")
                        continue
                    if not os.path.exists(arg):
                        console.print(f"[red]File not found: {arg}[/red]\n")
                        continue
                    with console.status(f"[bold green]Ingesting {arg}...[/bold green]"):
                        count = document_service.ingest_file(arg)
                    console.print(f"[green]✓ Ingested {count} chunks from {arg}. Total in DB: {vector_service().count()}[/green]\n")
                    continue

                elif cmd == "/stats":
                    total = vector_service().count()
                    console.print(f"[bold green]Total Indexed Chunks in ChromaDB:[/bold green] [bold white]{total}[/bold white]\n")
                    continue

                elif cmd == "/help":
                    console.print(
                        "[bold cyan]Available Commands:[/bold cyan]\n"
                        "  [bold green]/seed[/bold green]              - Re-indexes all docs in data/sample_docs/\n"
                        "  [bold green]/ingest <file>[/bold green]     - Ingests a specific file (.txt, .pdf, .md, .json)\n"
                        "  [bold green]/stats[/bold green]             - Shows total chunks in vector database\n"
                        "  [bold green]exit[/bold green]               - Quits the CLI\n"
                    )
                    continue

            with console.status("[bold green]Executing Self-Correcting RAG Workflow...[/bold green]", spinner="dots"):
                response = rag_service.process_query(query=query, thread_id=thread_id)

            # Display Execution Traces
            trace_table = Table(title="Agentic Execution Trace", border_style="dim")
            trace_table.add_column("Node", style="cyan", no_wrap=True)
            trace_table.add_column("Action", style="green")
            trace_table.add_column("Details", style="white")

            for step in response.trace_steps:
                details_str = ", ".join(f"{k}: {v}" for k, v in step.details.items())
                trace_table.add_row(step.node, step.action, details_str[:120])

            console.print(trace_table)

            # Display Summary Metrics
            console.print(
                f"[dim]Route: [bold]{response.route_taken}[/bold] | "
                f"Retries: [bold]{response.retries_count}[/bold] | "
                f"Grounding: [bold]{response.hallucination_check or 'N/A'}[/bold] | "
                f"Quality: [bold]{response.answer_quality_check or 'N/A'}[/bold][/dim]\n"
            )

            # Display Final Answer
            console.print(
                Panel(
                    Markdown(response.answer),
                    title=f"[bold green]Answer (Thread: {thread_id})[/bold green]",
                    border_style="green",
                )
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    interactive_repl()
