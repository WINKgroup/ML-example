from rich.progress import *
from rich.style import Style
from rich.console import Console

def get_pbar(console: Console) -> Progress:
    pbar = Progress(
        SpinnerColumn(spinner_name="monkey"),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[bold][progress.percentage]{task.percentage:>3.2f}%"),
        BarColumn(finished_style=Style(color="#008000")),
        MofNCompleteColumn(),
        TextColumn("[bold]•"),
        TimeElapsedColumn(),
        TextColumn("[bold]•"),
        TimeRemainingColumn(),
        TextColumn("[bold #5B4328]{task.speed} it/s"),
        SpinnerColumn(spinner_name="moon")
    )

    return pbar


def get_tasks(
    pbar: Progress, 
    epochs: int, 
    train_batches: int, 
    val_batches: int, 
    test_batches: int
) -> dict:
    
    tasks = {
        "epoch": pbar.add_task("[bold red]Epoch...", total=epochs),
        "train": pbar.add_task("[bold cyan]Train batches...", total=train_batches),
        "val": pbar.add_task("[bold yellow]Val batches...", total=val_batches),
        "test": pbar.add_task("[bold green]Test batches...", total=test_batches)
    }

    return tasks