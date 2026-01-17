"""Command-line interface for AutoML."""

import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from automl_tabular import run_automl_job
from automl_tabular.config import load_default_config

app = typer.Typer(
    name="automl",
    help="🤖 Explainable AutoML for Tabular Data",
    add_completion=False
)

console = Console()


@app.command()
def run(
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to CSV data file"
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Name of target column"
    ),
    output: str = typer.Option(
        "output",
        "--output",
        "-o",
        help="Output directory for models and reports"
    ),
    metric: str = typer.Option(
        "auto",
        "--metric",
        "-m",
        help="Primary metric (auto, roc_auc, accuracy, rmse, r2, etc.)"
    ),
    report: Optional[str] = typer.Option(
        None,
        "--report",
        "-r",
        help="Custom report filename"
    ),
    time_limit: Optional[int] = typer.Option(
        None,
        "--time-limit",
        help="Time limit in seconds for model search"
    ),
    max_trials: Optional[int] = typer.Option(
        None,
        "--max-trials",
        help="Maximum trials per model"
    ),
    algorithms: Optional[str] = typer.Option(
        None,
        "--algorithms",
        "-a",
        help="Comma-separated list of algorithms (e.g., 'logistic_regression,random_forest')"
    )
):
    """
    Run AutoML on a dataset.
    
    Example:
        automl run --data data.csv --target label
        automl run --data data.csv --target label --algorithms logistic_regression,random_forest
    """
    # Validate inputs
    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)
    
    # Build config overrides
    config = {}
    if time_limit is not None:
        config['search'] = {'time_limit_seconds': time_limit}
    if max_trials is not None:
        if 'search' not in config:
            config['search'] = {}
        config['search']['max_trials_per_model'] = max_trials
    if algorithms is not None:
        # Parse comma-separated algorithm list
        algo_list = [a.strip() for a in algorithms.split(',')]
        if 'models' not in config:
            config['models'] = {}
        config['models']['algorithms'] = algo_list
    
    # Show header
    console.print(Panel.fit(
        "🤖 [bold blue]AutoML - Explainable AutoML for Tabular Data[/bold blue]",
        border_style="blue"
    ))
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  📁 Data: {data}")
    console.print(f"  🎯 Target: {target}")
    console.print(f"  📊 Metric: {metric}")
    console.print(f"  📂 Output: {output}")
    if algorithms:
        console.print(f"  🔧 Algorithms: {algorithms}")
    
    try:
        # Run AutoML
        results = run_automl_job(
            data_path=str(data_path),
            target_column=target,
            output_dir=output,
            config=config if config else None,
            metric=metric,
            report_name=report
        )
        
        # Show results summary
        console.print("\n[bold green]✅ AutoML Completed Successfully![/bold green]\n")
        
        # Results table
        table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Problem Type", results['problem_type'].title())
        table.add_row("Best Model", results['best_model_name'])
        table.add_row(f"{results['metric_name'].upper()}", f"{results['metric_value']:.4f}")
        table.add_row("Models Tried", str(len(results['leaderboard'])))
        
        console.print(table)
        
        console.print(f"\n[bold]📁 Outputs:[/bold]")
        console.print(f"  💾 Model: [cyan]{results['model_path']}[/cyan]")
        console.print(f"  📄 Report: [cyan]{results['report_path']}[/cyan]")
        
        console.print(f"\n[dim]💡 Open the HTML report in your browser to see detailed results and visualizations.[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}\n")
        raise typer.Exit(1)


@app.command()
def predict(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to saved model file (.joblib)"
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to CSV data file for prediction"
    ),
    output: str = typer.Option(
        "predictions.csv",
        "--output",
        "-o",
        help="Output file for predictions"
    )
):
    """
    Make predictions using a trained model.
    
    Example:
        automl predict --model output/models/model.joblib --data new_data.csv
    """
    import joblib
    import pandas as pd
    
    # Validate inputs
    model_path = Path(model)
    data_path = Path(data)
    
    if not model_path.exists():
        console.print(f"[red]Error: Model file not found: {model}[/red]")
        raise typer.Exit(1)
    
    if not data_path.exists():
        console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        "🔮 [bold blue]Making Predictions[/bold blue]",
        border_style="blue"
    ))
    
    try:
        # Load model
        console.print("\n📥 Loading model...")
        pipeline = joblib.load(model_path)
        
        # Load data
        console.print("📁 Loading data...")
        df = pd.read_csv(data_path)
        console.print(f"   Loaded {len(df)} samples")
        
        # Make predictions
        console.print("🔮 Generating predictions...")
        predictions = pipeline.predict(df)
        
        # Save predictions
        output_df = df.copy()
        output_df['prediction'] = predictions
        
        # Add probabilities if available
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(df)
            if proba.shape[1] == 2:
                output_df['probability'] = proba[:, 1]
            else:
                for i in range(proba.shape[1]):
                    output_df[f'probability_class_{i}'] = proba[:, i]
        
        output_df.to_csv(output, index=False)
        
        console.print(f"\n[bold green]✅ Predictions saved to:[/bold green] [cyan]{output}[/cyan]\n")
        
        # Show sample predictions
        console.print("[bold]Sample predictions:[/bold]")
        console.print(output_df.head(10).to_string(index=False))
        console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}\n")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from automl_tabular import __version__
    console.print(f"\n[bold blue]AutoML Tabular[/bold blue] version [green]{__version__}[/green]\n")


@app.command()
def info():
    """Show configuration information."""
    config = load_default_config()
    
    console.print(Panel.fit(
        "ℹ️  [bold blue]AutoML Configuration[/bold blue]",
        border_style="blue"
    ))
    
    console.print("\n[bold]Experiment Settings:[/bold]")
    for key, value in config['experiment'].items():
        console.print(f"  {key}: {value}")
    
    console.print("\n[bold]Search Settings:[/bold]")
    for key, value in config['search'].items():
        console.print(f"  {key}: {value}")
    
    console.print("\n[bold]Available Classifiers:[/bold]")
    for model in config['models']['classifiers']:
        status = "✓" if model.get('enabled', True) else "✗"
        console.print(f"  [{status}] {model['name']}")
    
    console.print("\n[bold]Available Regressors:[/bold]")
    for model in config['models']['regressors']:
        status = "✓" if model.get('enabled', True) else "✗"
        console.print(f"  [{status}] {model['name']}")
    
    console.print()


def main():
    """Main entry point for CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
