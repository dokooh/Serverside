#!/usr/bin/env python3
"""
Report Generator and Visualization Script
Creates comprehensive reports with charts comparing model performance
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

logger = logging.getLogger(__name__)

class ModelPerformanceReporter:
    """Generates comprehensive performance reports with visualizations"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure plotly
        self.plotly_theme = "plotly_white"
    
    def load_benchmark_data(self, benchmark_file: str, summary_file: str = None) -> tuple:
        """Load benchmark data from JSON files"""
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        
        summary_data = None
        if summary_file and Path(summary_file).exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
        
        return benchmark_data, summary_data
    
    def create_performance_dataframe(self, benchmark_data: List[Dict]) -> pd.DataFrame:
        """Convert benchmark data to pandas DataFrame for analysis"""
        rows = []
        
        for result in benchmark_data:
            if result.get('error'):
                continue  # Skip failed results
            
            # Extract model name (remove prompt suffix)
            model_name = result['model_name']
            if '_prompt_' in model_name:
                model_name = model_name.split('_prompt_')[0]
            
            row = {
                'model': model_name,
                'full_name': result['model_name'],
                'prompt': result['prompt'][:50] + '...' if len(result['prompt']) > 50 else result['prompt'],
                'response_time': result['response_time_seconds'],
                'tokens_generated': result['tokens_generated'],
                'tokens_per_second': result['tokens_per_second'],
                'peak_memory_gb': result['peak_memory_gb'],
                'avg_memory_gb': result['avg_memory_gb'],
                'peak_gpu_memory_gb': result['peak_gpu_memory_gb'],
                'avg_gpu_memory_gb': result['avg_gpu_memory_gb'],
                'avg_gpu_utilization': result['avg_gpu_utilization'],
                'max_gpu_temperature': result['max_gpu_temperature'],
                'response_length': len(result['response']),
                'success': result.get('error') is None
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_model_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive model comparison chart"""
        # Aggregate by model
        model_stats = df.groupby('model').agg({
            'response_time': 'mean',
            'tokens_per_second': 'mean',
            'peak_memory_gb': 'mean',
            'peak_gpu_memory_gb': 'mean',
            'avg_gpu_utilization': 'mean'
        }).round(2)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Average Response Time (s)', 'Tokens per Second', 'Peak Memory Usage (GB)',
                'Peak GPU Memory (GB)', 'GPU Utilization (%)', 'Model Overview'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        models = model_stats.index.tolist()
        colors = px.colors.qualitative.Set3[:len(models)]
        
        # Response time (lower is better)
        fig.add_trace(
            go.Bar(x=models, y=model_stats['response_time'], 
                   marker_color=colors, name='Response Time'),
            row=1, col=1
        )
        
        # Tokens per second (higher is better)
        fig.add_trace(
            go.Bar(x=models, y=model_stats['tokens_per_second'], 
                   marker_color=colors, name='Tokens/sec'),
            row=1, col=2
        )
        
        # Memory usage
        fig.add_trace(
            go.Bar(x=models, y=model_stats['peak_memory_gb'], 
                   marker_color=colors, name='Peak Memory'),
            row=1, col=3
        )
        
        # GPU Memory
        fig.add_trace(
            go.Bar(x=models, y=model_stats['peak_gpu_memory_gb'], 
                   marker_color=colors, name='GPU Memory'),
            row=2, col=1
        )
        
        # GPU Utilization
        fig.add_trace(
            go.Bar(x=models, y=model_stats['avg_gpu_utilization'], 
                   marker_color=colors, name='GPU Util %'),
            row=2, col=2
        )
        
        # Summary table
        table_data = []
        for model in models:
            stats = model_stats.loc[model]
            table_data.append([
                model,
                f"{stats['response_time']:.2f}s",
                f"{stats['tokens_per_second']:.1f}",
                f"{stats['peak_memory_gb']:.1f}GB",
                f"{stats['avg_gpu_utilization']:.1f}%"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'Avg Time', 'Tokens/s', 'Peak RAM', 'GPU %']),
                cells=dict(values=list(zip(*table_data)))
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            template=self.plotly_theme,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_detailed_metrics_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create detailed metrics visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Response Time vs Tokens Generated',
                'Memory Efficiency',
                'GPU Utilization Distribution',
                'Performance by Prompt Type'
            ]
        )
        
        models = df['model'].unique()
        colors = px.colors.qualitative.Set3[:len(models)]
        color_map = dict(zip(models, colors))
        
        # Scatter: Response time vs tokens
        for model in models:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['tokens_generated'],
                    y=model_data['response_time'],
                    mode='markers',
                    name=model,
                    marker=dict(color=color_map[model]),
                    text=model_data['prompt'],
                    hovertemplate='<b>%{text}</b><br>Tokens: %{x}<br>Time: %{y:.2f}s<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Memory efficiency scatter
        for model in models:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['peak_memory_gb'],
                    y=model_data['tokens_per_second'],
                    mode='markers',
                    name=model,
                    marker=dict(color=color_map[model]),
                    showlegend=False,
                    text=model_data['prompt'],
                    hovertemplate='<b>%{text}</b><br>Memory: %{x:.1f}GB<br>Speed: %{y:.1f} tok/s<extra></extra>'
                ),
                row=1, col=2
            )
        
        # GPU utilization histogram
        for model in models:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Histogram(
                    x=model_data['avg_gpu_utilization'],
                    name=model,
                    marker=dict(color=color_map[model]),
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Performance by prompt (box plot)
        fig.add_trace(
            go.Box(
                x=df['model'],
                y=df['tokens_per_second'],
                marker=dict(color='lightblue'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Tokens Generated", row=1, col=1)
        fig.update_yaxes(title_text="Response Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Peak Memory (GB)", row=1, col=2)
        fig.update_yaxes(title_text="Tokens per Second", row=1, col=2)
        fig.update_xaxes(title_text="GPU Utilization (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Tokens per Second", row=2, col=2)
        
        fig.update_layout(
            title_text="Detailed Performance Metrics",
            template=self.plotly_theme,
            height=800
        )
        
        return fig
    
    def create_resource_utilization_timeline(self, benchmark_data: List[Dict]) -> go.Figure:
        """Create resource utilization timeline for a sample model"""
        # Find a model with good resource data
        sample_result = None
        for result in benchmark_data:
            if result.get('resource_snapshots') and len(result['resource_snapshots']) > 10:
                sample_result = result
                break
        
        if not sample_result:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(text="No resource timeline data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        snapshots = sample_result['resource_snapshots']
        
        # Convert to relative time
        start_time = snapshots[0]['timestamp']
        times = [(s['timestamp'] - start_time) for s in snapshots]
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Memory Usage', 'GPU Memory', 'GPU Utilization'],
            shared_xaxes=True
        )
        
        # Memory usage
        memory_used = [s['memory_used_gb'] for s in snapshots]
        fig.add_trace(
            go.Scatter(x=times, y=memory_used, mode='lines', name='RAM Used (GB)', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # GPU Memory
        gpu_memory = [s['gpu_memory_used_gb'] for s in snapshots]
        fig.add_trace(
            go.Scatter(x=times, y=gpu_memory, mode='lines', name='GPU Memory (GB)', 
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # GPU Utilization
        gpu_util = [s['gpu_utilization_percent'] for s in snapshots]
        fig.add_trace(
            go.Scatter(x=times, y=gpu_util, mode='lines', name='GPU Util (%)', 
                      line=dict(color='green')),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
        fig.update_yaxes(title_text="GPU Memory (GB)", row=2, col=1)
        fig.update_yaxes(title_text="GPU Utilization (%)", row=3, col=1)
        
        fig.update_layout(
            title_text=f"Resource Timeline - {sample_result['model_name']}",
            template=self.plotly_theme,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_model_efficiency_radar(self, df: pd.DataFrame) -> go.Figure:
        """Create radar chart for model efficiency metrics"""
        # Normalize metrics for radar chart (0-100 scale)
        model_stats = df.groupby('model').agg({
            'tokens_per_second': 'mean',
            'response_time': 'mean',
            'peak_memory_gb': 'mean',
            'peak_gpu_memory_gb': 'mean',
            'avg_gpu_utilization': 'mean'
        })
        
        # Normalize metrics (higher is better for all)
        normalized_stats = model_stats.copy()
        
        # For metrics where lower is better, invert them
        normalized_stats['response_time'] = 100 / (model_stats['response_time'] + 0.1)  # Avoid division by zero
        normalized_stats['peak_memory_gb'] = 100 / (model_stats['peak_memory_gb'] + 0.1)
        normalized_stats['peak_gpu_memory_gb'] = 100 / (model_stats['peak_gpu_memory_gb'] + 0.1)
        
        # Scale all to 0-100
        for col in normalized_stats.columns:
            max_val = normalized_stats[col].max()
            normalized_stats[col] = (normalized_stats[col] / max_val) * 100
        
        fig = go.Figure()
        
        categories = ['Speed', 'Response Time', 'Memory Efficiency', 'GPU Memory Eff.', 'GPU Utilization']
        
        for model in normalized_stats.index:
            values = [
                normalized_stats.loc[model, 'tokens_per_second'],
                normalized_stats.loc[model, 'response_time'],
                normalized_stats.loc[model, 'peak_memory_gb'],
                normalized_stats.loc[model, 'peak_gpu_memory_gb'],
                normalized_stats.loc[model, 'avg_gpu_utilization']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill='toself',
                name=model,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Model Efficiency Comparison (Normalized)",
            template=self.plotly_theme
        )
        
        return fig
    
    def generate_html_report(self, benchmark_file: str, summary_file: str = None) -> str:
        """Generate comprehensive HTML report"""
        logger.info("Generating comprehensive HTML report...")
        
        # Load data
        benchmark_data, summary_data = self.load_benchmark_data(benchmark_file, summary_file)
        df = self.create_performance_dataframe(benchmark_data)
        
        if df.empty:
            logger.warning("No valid benchmark data found")
            return ""
        
        # Create all charts
        comparison_chart = self.create_model_comparison_chart(df)
        detailed_chart = self.create_detailed_metrics_chart(df)
        timeline_chart = self.create_resource_utilization_timeline(benchmark_data)
        radar_chart = self.create_model_efficiency_radar(df)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Performance Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .chart-container {{ margin: 20px 0; height: 800px; }}
                .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .summary-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hugging Face Model Performance Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <table class="summary-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Models Tested</td><td>{df['model'].nunique()}</td></tr>
                    <tr><td>Total Benchmark Runs</td><td>{len(df)}</td></tr>
                    <tr><td>Average Response Time</td><td>{df['response_time'].mean():.2f} seconds</td></tr>
                    <tr><td>Average Tokens/Second</td><td>{df['tokens_per_second'].mean():.1f}</td></tr>
                    <tr><td>Average Memory Usage</td><td>{df['peak_memory_gb'].mean():.1f} GB</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <div id="comparison-chart" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Detailed Performance Metrics</h2>
                <div id="detailed-chart" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Model Efficiency Radar</h2>
                <div id="radar-chart" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Resource Utilization Timeline</h2>
                <div id="timeline-chart" class="chart-container"></div>
            </div>
            
            <script>
                Plotly.newPlot('comparison-chart', {comparison_chart.to_json()});
                Plotly.newPlot('detailed-chart', {detailed_chart.to_json()});
                Plotly.newPlot('radar-chart', {radar_chart.to_json()});
                Plotly.newPlot('timeline-chart', {timeline_chart.to_json()});
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = self.output_dir / "performance_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")
        return str(html_file)
    
    def create_static_charts(self, benchmark_file: str) -> List[str]:
        """Create static PNG charts using matplotlib"""
        logger.info("Creating static charts...")
        
        benchmark_data, _ = self.load_benchmark_data(benchmark_file)
        df = self.create_performance_dataframe(benchmark_data)
        
        if df.empty:
            logger.warning("No valid benchmark data found")
            return []
        
        chart_files = []
        
        # Model comparison bar chart
        plt.figure(figsize=(12, 8))
        model_stats = df.groupby('model').agg({
            'response_time': 'mean',
            'tokens_per_second': 'mean',
            'peak_memory_gb': 'mean'
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Response time
        model_stats['response_time'].plot(kind='bar', ax=axes[0,0], title='Average Response Time (s)')
        axes[0,0].set_ylabel('Seconds')
        
        # Tokens per second
        model_stats['tokens_per_second'].plot(kind='bar', ax=axes[0,1], title='Tokens per Second')
        axes[0,1].set_ylabel('Tokens/sec')
        
        # Memory usage
        model_stats['peak_memory_gb'].plot(kind='bar', ax=axes[1,0], title='Peak Memory Usage (GB)')
        axes[1,0].set_ylabel('Memory (GB)')
        
        # Scatter plot: performance vs memory
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            axes[1,1].scatter(model_data['peak_memory_gb'], model_data['tokens_per_second'], 
                            label=model, alpha=0.7)
        axes[1,1].set_xlabel('Peak Memory (GB)')
        axes[1,1].set_ylabel('Tokens per Second')
        axes[1,1].set_title('Performance vs Memory Usage')
        axes[1,1].legend()
        
        plt.tight_layout()
        chart_file = self.output_dir / "model_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file))
        
        logger.info(f"Static charts saved: {len(chart_files)} files")
        return chart_files

def main():
    """Main function to generate reports"""
    parser = argparse.ArgumentParser(description="Generate model performance reports")
    parser.add_argument("--benchmark-file", required=True, help="JSON file with benchmark results")
    parser.add_argument("--summary-file", help="JSON file with summary results")
    parser.add_argument("--output-dir", default="./reports", help="Output directory for reports")
    parser.add_argument("--format", choices=["html", "static", "both"], default="both", 
                       help="Report format to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check input files
    if not Path(args.benchmark_file).exists():
        logger.error(f"Benchmark file not found: {args.benchmark_file}")
        return
    
    # Initialize reporter
    reporter = ModelPerformanceReporter(output_dir=args.output_dir)
    
    try:
        if args.format in ["html", "both"]:
            html_file = reporter.generate_html_report(args.benchmark_file, args.summary_file)
            print(f"HTML report: {html_file}")
        
        if args.format in ["static", "both"]:
            chart_files = reporter.create_static_charts(args.benchmark_file)
            print(f"Static charts: {chart_files}")
        
        print(f"Reports saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

if __name__ == "__main__":
    main()