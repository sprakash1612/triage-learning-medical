#!/usr/bin/env python3
"""
Generate comprehensive HTML report from experiment results.

Aggregates results from experiment directory and creates
interactive HTML report with executive summary, metrics,
visualizations, and analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_json_file(filepath: Path) -> Dict:
    """Safely load JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load {filepath}: {e}")
        return {}


def load_numpy_file(filepath: Path) -> Optional[np.ndarray]:
    """Safely load numpy file."""
    try:
        return np.load(filepath)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load {filepath}: {e}")
        return None


def extract_experiment_results(experiment_dir: Path) -> Dict:
    """
    Extract all results from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
    
    Returns:
        Dictionary with aggregated results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'config': load_json_file(experiment_dir / 'config.yaml'),
        'metrics': {},
        'visualizations': [],
    }
    
    # Look for standard result files
    results_subdir = experiment_dir / 'results'
    if results_subdir.exists():
        # Training metrics
        if (results_subdir / 'training_metrics.json').exists():
            results['training_metrics'] = load_json_file(results_subdir / 'training_metrics.json')
        
        # Evaluation metrics
        if (results_subdir / 'metrics.json').exists():
            results['metrics']['evaluation'] = load_json_file(results_subdir / 'metrics.json')
        
        # Uncertainty analysis
        if (results_subdir / 'uncertainty' / 'metrics.json').exists():
            results['metrics']['uncertainty'] = load_json_file(
                results_subdir / 'uncertainty' / 'metrics.json'
            )
        
        # Triage analysis
        if (results_subdir / 'triage' / 'triage_report.txt').exists():
            with open(results_subdir / 'triage' / 'triage_report.txt') as f:
                results['triage_report'] = f.read()
        
        # Collect visualizations
        for png_file in results_subdir.glob('**/*.png'):
            results['visualizations'].append(str(png_file.relative_to(experiment_dir)))
    
    # Look for logs
    logs_dir = experiment_dir / 'logs'
    if logs_dir.exists():
        log_files = list(logs_dir.glob('*.log'))
        if log_files:
            results['log_file'] = str(log_files[0].relative_to(experiment_dir))
    
    return results


def create_html_report(
    experiment_results: Dict,
    output_path: Path,
    include_plots: bool = True,
) -> None:
    """
    Create comprehensive HTML report.
    
    Args:
        experiment_results: Dictionary with experiment results
        output_path: Path to save HTML report
        include_plots: Whether to embed plots
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - Triage Learning System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        section {
            margin-bottom: 40px;
        }
        
        h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .metric-card h4 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background: #f5f5f5;
            font-weight: 600;
            color: #667eea;
        }
        
        .metrics-table tr:hover {
            background: #f9f9f9;
        }
        
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .visualization-caption {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
            font-style: italic;
        }
        
        .report-text {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }
        
        .summary-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .summary-box h3 {
            margin-top: 0;
        }
        
        .badge {
            display: inline-block;
            padding: 5px 10px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            font-size: 0.8em;
            margin: 5px;
        }
        
        .badge.success { background: #28a745; }
        .badge.warning { background: #ffc107; }
        .badge.error { background: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Triage Learning System Report</h1>
            <p>Comprehensive Evaluation and Analysis</p>
        </header>
        
        <div class="content">
"""
    
    # Executive Summary
    html_content += """
            <section>
                <h2>Executive Summary</h2>
                <div class="summary-box">
"""
    
    if 'metrics' in experiment_results and 'evaluation' in experiment_results['metrics']:
        metrics = experiment_results['metrics']['evaluation']
        html_content += f"""
                    <p>
                        <strong>Model Accuracy:</strong> {metrics.get('accuracy', 'N/A'):.4f}<br>
                        <strong>Balanced Accuracy:</strong> {metrics.get('balanced_accuracy', 'N/A'):.4f}<br>
                        <strong>F1 Score:</strong> {metrics.get('f1_score', 'N/A'):.4f}
                    </p>
"""
    
    html_content += f"""
                    <p><strong>Report Generated:</strong> {experiment_results['timestamp']}</p>
                    <p><strong>Experiment Directory:</strong> {experiment_results['experiment_dir']}</p>
                </div>
            </section>
"""
    
    # Evaluation Metrics
    if 'metrics' in experiment_results and 'evaluation' in experiment_results['metrics']:
        html_content += """
            <section>
                <h2>Evaluation Metrics</h2>
"""
        metrics = experiment_results['metrics']['evaluation']
        
        # Create metrics table
        html_content += """
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                html_content += f"                        <tr><td>{key}</td><td>{value:.6f}</td></tr>\n"
            else:
                html_content += f"                        <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html_content += """
                    </tbody>
                </table>
            </section>
"""
    
    # Uncertainty Analysis
    if 'metrics' in experiment_results and 'uncertainty' in experiment_results['metrics']:
        html_content += """
            <section>
                <h2>Uncertainty Analysis</h2>
"""
        metrics = experiment_results['metrics']['uncertainty']
        
        html_content += """
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                html_content += f"                        <tr><td>{key}</td><td>{value:.6f}</td></tr>\n"
            else:
                html_content += f"                        <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html_content += """
                    </tbody>
                </table>
            </section>
"""
    
    # Triage Report
    if 'triage_report' in experiment_results:
        html_content += f"""
            <section>
                <h2>Triage Analysis</h2>
                <div class="report-text">{experiment_results['triage_report']}</div>
            </section>
"""
    
    # Visualizations
    if include_plots and 'visualizations' in experiment_results and experiment_results['visualizations']:
        html_content += """
            <section>
                <h2>Visualizations</h2>
"""
        for viz in experiment_results['visualizations']:
            # Create relative path for embedding
            viz_name = Path(viz).stem.replace('_', ' ').title()
            html_content += f"""
                <div class="visualization">
                    <img src="{viz}" alt="{viz_name}">
                    <div class="visualization-caption">{viz_name}</div>
                </div>
"""
        
        html_content += """
            </section>
"""
    
    # Footer
    html_content += """
        </div>
        
        <footer>
            <p>Generated by Triage Learning System | """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>For questions or issues, please refer to the documentation.</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    logger.info(f"HTML report saved to {output_path}")


def export_to_pdf(
    html_path: Path,
    pdf_path: Path,
) -> bool:
    """
    Export HTML report to PDF.
    
    Args:
        html_path: Path to HTML file
        pdf_path: Path to save PDF
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from weasyprint import HTML
        HTML(str(html_path)).write_pdf(str(pdf_path))
        logger.info(f"PDF report saved to {pdf_path}")
        return True
    except ImportError:
        logger.warning("WeasyPrint not installed. Install with: pip install weasyprint")
        return False
    except Exception as e:
        logger.error(f"Failed to export PDF: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive experiment report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report
  python generate_report.py --experiment-dir experiments/exp_001
  
  # Generate HTML and PDF
  python generate_report.py --experiment-dir experiments/exp_001 --pdf
  
  # Custom output location
  python generate_report.py --experiment-dir experiments/exp_001 --output-file results/report.html
        """,
    )
    
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Path to experiment directory',
    )
    parser.add_argument(
        '--output-file',
        help='Path to save HTML report (default: experiment_dir/report.html)',
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Also export to PDF',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Exclude embedded plots',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load experiment results
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    logger.info(f"Loading results from {experiment_dir}")
    experiment_results = extract_experiment_results(experiment_dir)
    
    # Determine output path
    if args.output_file:
        output_html = Path(args.output_file)
    else:
        output_html = experiment_dir / 'report.html'
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    create_html_report(
        experiment_results,
        output_html,
        include_plots=not args.no_plots,
    )
    
    # Export to PDF if requested
    if args.pdf:
        output_pdf = output_html.with_suffix('.pdf')
        export_to_pdf(output_html, output_pdf)
    
    logger.info("Report generation complete!")


if __name__ == '__main__':
    main()
