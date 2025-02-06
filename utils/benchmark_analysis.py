"""
Benchmark analysis utilities for processing benchmark data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class BenchmarkAnalyzer:
    """Analyzes benchmark data from JSONL files."""
    
    def __init__(self, benchmark_dir: str = "benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)
        
        # Use the specified color scheme
        self.colors = ['#fee8c8', '#fdbb84', '#e34a33']
        
        # Map operations to the color scheme
        self.operation_colors = {
            'inference': '#fee8c8',
            'process': '#fdbb84',
            'finalize': '#e34a33',
            'capture': '#fee8c8'  # Reuse first color if needed
        }
        
        # Set dark theme with consistent colors
        plt.style.use('dark_background')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.facecolor': '#0C0C0C',
            'axes.facecolor': '#0C0C0C',
            'grid.alpha': 0.3,
            'grid.color': '#fee8c8',  # Use lightest color for grid
            'savefig.facecolor': '#0C0C0C',
            'text.color': '#fdbb84',  # Use middle color for text
            'axes.labelcolor': '#fdbb84',
            'xtick.color': '#fdbb84',
            'ytick.color': '#fdbb84',
            'axes.edgecolor': '#fee8c8'  # Use lightest color for edges
        })
        
        # Warmup period (skip first N samples for better visualization)
        self.warmup_samples = 50
        
        # Units for different metrics
        self.metric_units = {
            'duration_ms': 'ms',
            'throughput': 'samples/sec',
            'memory_mb': 'MB',
            'data_size': 'samples'
        }
    
    def _get_operation_color(self, operation: str) -> str:
        """Get consistent color for an operation."""
        return self.operation_colors.get(operation, '#fdbb84')  # Default to middle color
    
    def _get_operation_colors(self, operations: List[str]) -> List[str]:
        """Get list of colors for operations."""
        # Cycle through the color scheme if more operations than colors
        return [self.colors[i % len(self.colors)] for i in range(len(operations))]
    
    def _format_axis_label(self, metric: str) -> str:
        """Format axis label with units."""
        unit = self.metric_units.get(metric, '')
        if unit:
            return f"{metric.replace('_', ' ').title()} ({unit})"
        return metric.replace('_', ' ').title()
    
    def load_component_data(self, component: str, latest_only: bool = True) -> pd.DataFrame:
        """Load benchmark data for a component."""
        component_dir = self.benchmark_dir / component
        if not component_dir.exists():
            raise ValueError(f"No benchmark data found for component: {component}")
        
        # Get all benchmark files
        benchmark_files = sorted(component_dir.glob("benchmark_*.jsonl"))
        if not benchmark_files:
            raise ValueError(f"No benchmark files found in {component_dir}")
        
        # Use only the latest file if requested
        if latest_only:
            benchmark_files = benchmark_files[-1:]
        
        data = []
        for file in benchmark_files:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # For metric entries, flatten the data field
                        if entry['type'] == 'metric' and isinstance(entry.get('data'), dict):
                            metric_data = entry['data'].copy()
                            del entry['data']
                            entry.update(metric_data)
                        data.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return pd.DataFrame(data)
    
    def analyze_metrics(self, component: str) -> Dict[str, Any]:
        """Analyze metrics for a component."""
        df = self.load_component_data(component)
        
        # Get system info from header
        system_info = {}
        header_rows = df[df['type'] == 'header']
        if not header_rows.empty and 'system_info' in header_rows.iloc[0]:
            system_info = header_rows.iloc[0]['system_info']
        
        # Get metrics
        metrics_df = df[df['type'] == 'metric'].copy()
        
        analysis = {
            'system_info': system_info,
            'metrics_summary': {},
            'error_summary': {}
        }
        
        if not metrics_df.empty:
            # Calculate metrics per operation
            for operation in metrics_df['operation'].unique():
                op_metrics = metrics_df[metrics_df['operation'] == operation]
                summary = {
                    'count': len(op_metrics),
                    'duration_ms': {
                        'mean': op_metrics['duration_ms'].mean(),
                        'median': op_metrics['duration_ms'].median(),
                        'min': op_metrics['duration_ms'].min(),
                        'max': op_metrics['duration_ms'].max(),
                        'std': op_metrics['duration_ms'].std()
                    },
                    'throughput': {
                        'mean': op_metrics['throughput'].mean(),
                        'median': op_metrics['throughput'].median(),
                        'min': op_metrics['throughput'].min(),
                        'max': op_metrics['throughput'].max()
                    }
                }
                
                # Add memory stats if available
                if 'memory_mb' in op_metrics.columns:
                    summary['memory_mb'] = {
                        'mean': op_metrics['memory_mb'].mean(),
                        'max': op_metrics['memory_mb'].max(),
                        'min': op_metrics['memory_mb'].min()
                    }
                
                # Add context stats if available
                if 'context' in op_metrics.columns:
                    contexts = [c for c in op_metrics['context'] if isinstance(c, dict)]
                    if contexts:
                        context_df = pd.DataFrame(contexts)
                        for col in context_df.columns:
                            if pd.api.types.is_numeric_dtype(context_df[col]):
                                summary[f'context_{col}'] = {
                                    'mean': context_df[col].mean(),
                                    'max': context_df[col].max(),
                                    'min': context_df[col].min()
                                }
                
                analysis['metrics_summary'][operation] = summary
        
        # Get errors
        errors_df = df[df['type'] == 'event'].copy()
        if not errors_df.empty:
            error_counts = errors_df.groupby('event_type')['timestamp'].count()
            analysis['error_summary'] = error_counts.to_dict()
        
        return analysis
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by removing warmup period and extreme outliers."""
        if df.empty:
            return df
            
        # Skip warmup period - use smaller warmup for small datasets
        warmup = min(5, len(df) // 10) if len(df) < 100 else self.warmup_samples
        df = df.iloc[warmup:]
        
        # For each numeric column, clip extreme outliers at 99th percentile
        numeric_cols = ['duration_ms', 'throughput', 'memory_mb']
        for col in numeric_cols:
            if col in df.columns and not df[col].empty:
                q99 = df[col].quantile(0.99)
                if not pd.isna(q99):
                    df[col] = df[col].clip(upper=q99)
        
        return df
    
    def _plot_duration_distributions(self, df, component, plot_dir):
        """Generate duration distribution plots."""
        try:
            if len(df) < 2:
                logger.warning(f"Not enough data points for {component} duration plots")
                return
                
            plt.figure(figsize=(15, 8))
            operations = df['operation'].unique()
            
            # 1. Duration Over Time
            plt.subplot(2, 1, 1)
            for operation in operations:
                op_data = df[df['operation'] == operation]
                if len(op_data) > 1:
                    plt.plot(op_data.index, op_data['duration_ms'],
                           label=operation, alpha=0.6,
                           color=self._get_operation_color(operation))
            plt.title(f'{component.title()} Duration Over Time')
            plt.xlabel('Operation Index')
            plt.ylabel('Duration (ms)')
            plt.legend()
            
            # 2. Box Plot of Duration
            plt.subplot(2, 1, 2)
            sns.boxplot(data=df, x='operation', y='duration_ms',
                       palette=self.colors)
            plt.title('Duration Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Duration (ms)')
            
            # Save plot
            plt.tight_layout()
            output_path = os.path.join(plot_dir, f"{component}_duration.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting duration distributions: {e}")
            plt.close()
    
    def _plot_throughput_variations(self, df, component, plot_dir):
        """Generate throughput variation plots."""
        try:
            if len(df) < 2:
                logger.warning(f"Not enough data points for {component} throughput plots")
                return
                
            plt.figure(figsize=(15, 8))
            operations = df['operation'].unique()
            
            # 1. Throughput Over Time
            plt.subplot(2, 1, 1)
            for operation in operations:
                op_data = df[df['operation'] == operation]
                if len(op_data) > 1:
                    plt.plot(op_data.index, op_data['throughput'],
                           label=operation, alpha=0.6,
                           color=self._get_operation_color(operation))
            plt.title(f'{component.title()} Throughput Over Time')
            plt.xlabel('Operation Index')
            plt.ylabel('Throughput (samples/sec)')
            plt.legend()
            
            # 2. Box Plot of Throughput
            plt.subplot(2, 1, 2)
            sns.boxplot(data=df, x='operation', y='throughput',
                       palette=self.colors)
            plt.title('Throughput Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Throughput (samples/sec)')
            
            # Save plot
            plt.tight_layout()
            output_path = os.path.join(plot_dir, f"{component}_throughput.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting throughput variations: {e}")
            plt.close()
    
    def _plot_memory_usage(self, df, component, plot_dir):
        """Generate memory usage plots."""
        try:
            if len(df) < 2:
                logger.warning(f"Not enough data points for {component} memory plots")
                return
                
            plt.figure(figsize=(15, 8))
            operations = df['operation'].unique()
            
            # 1. Memory Usage Over Time
            plt.subplot(2, 1, 1)
            for operation in operations:
                op_data = df[df['operation'] == operation]
                if len(op_data) > 1:
                    plt.plot(op_data.index, op_data['memory_mb'],
                           label=operation, alpha=0.6,
                           color=self._get_operation_color(operation))
            plt.title(f'{component.title()} Memory Usage Over Time')
            plt.xlabel('Operation Index')
            plt.ylabel('Memory Usage (MB)')
            plt.legend()
            
            # 2. Box Plot of Memory Usage
            plt.subplot(2, 1, 2)
            sns.boxplot(data=df, x='operation', y='memory_mb',
                       palette=self.colors)
            plt.title('Memory Usage Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Memory Usage (MB)')
            
            # Save plot
            plt.tight_layout()
            output_path = os.path.join(plot_dir, f"{component}_memory.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting memory usage: {e}")
            plt.close()
    
    def _plot_correlation_heatmap(self, df, component, plot_dir):
        """Generate correlation heatmap with improved readability."""
        try:
            numeric_cols = ['duration_ms', 'throughput', 'memory_mb', 'data_size']
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation = df[numeric_cols].corr()
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(correlation, dtype=bool))
                
                # Create custom colormap from our color scheme
                custom_cmap = sns.color_palette([self.colors[0], '#0C0C0C', self.colors[2]], as_cmap=True)
                
                # Plot heatmap with improved annotations
                sns.heatmap(correlation, mask=mask, annot=True, cmap=custom_cmap,
                           center=0, fmt='.2f', square=True, linewidths=1,
                           cbar_kws={"shrink": .8},
                           annot_kws={"size": 10, "weight": "bold", "color": self.colors[1]})
                
                plt.title('Metric Correlations\n(1.0 = Perfect Correlation, -1.0 = Perfect Inverse Correlation)',
                         pad=20, color=self.colors[1])
                
                if plot_dir:
                    plt.savefig(plot_dir / f'{component}_correlation.png',
                               dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error plotting correlation heatmap: {str(e)}")
            plt.close()
    
    def _plot_metric_trends(self, df, component, plot_dir):
        """Generate combined metric trend plots."""
        try:
            plt.figure(figsize=(15, 10))
            operations = df['operation'].unique()
            
            # Create GridSpec with proper spacing
            gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            
            # Throughput
            ax1 = plt.subplot(gs[0])
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                ax1.plot(op_data.index, op_data['throughput'].rolling(window=20, min_periods=1).mean(),
                        color='#e34a33', label=operations[0], linewidth=2)
            else:
                colors = self._get_operation_colors(operations)
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if not op_data.empty:
                        ax1.plot(op_data.index, op_data['throughput'].rolling(window=20, min_periods=1).mean(),
                                color=colors[i], label=operation)
            ax1.set_title('Performance Overview')
            ax1.legend()
            ax1.set_ylabel('Throughput (samples/sec)')
            ax1.grid(True, alpha=0.3)
            
            # Duration
            ax2 = plt.subplot(gs[1])
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                ax2.plot(op_data.index, op_data['duration_ms'].rolling(window=20, min_periods=1).mean(),
                        color='#e34a33', label=operations[0], linewidth=2)
            else:
                colors = self._get_operation_colors(operations)
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if not op_data.empty:
                        ax2.plot(op_data.index, op_data['duration_ms'].rolling(window=20, min_periods=1).mean(),
                                color=colors[i], label=operation)
            ax2.set_ylabel('Duration (ms)')
            ax2.grid(True, alpha=0.3)
            
            # Memory if available
            if 'memory_mb' in df.columns:
                ax3 = plt.subplot(gs[2])
                if len(operations) == 1:
                    op_data = df[df['operation'] == operations[0]]
                    ax3.plot(op_data.index, op_data['memory_mb'].rolling(window=20, min_periods=1).mean(),
                            color='#e34a33', label=operations[0], linewidth=2)
                else:
                    colors = self._get_operation_colors(operations)
                    for i, operation in enumerate(operations):
                        op_data = df[df['operation'] == operation]
                        if not op_data.empty:
                            ax3.plot(op_data.index, op_data['memory_mb'].rolling(window=20, min_periods=1).mean(),
                                    color=colors[i], label=operation)
                ax3.set_ylabel('Memory (MB)')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if plot_dir:
                plt.savefig(plot_dir / f'{component}_trends.png',
                           dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting metric trends: {str(e)}")
            plt.close()
    
    def plot_metrics(self, component: str, save_dir: Optional[str] = None):
        """Generate plots for component metrics."""
        print(f"\nGenerating plots for {component}...")
        
        try:
            df = self.load_component_data(component)
            metrics_df = df[df['type'] == 'metric'].copy()
            
            if metrics_df.empty:
                print(f"No metrics data found for component: {component}")
                return
            
            # Prepare data (remove warmup period and outliers)
            metrics_df = self._prepare_data(metrics_df)
            
            if save_dir:
                plot_dir = Path(save_dir)
                plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure we have enough colors for operations
            n_ops = len(metrics_df['operation'].unique())
            if n_ops > len(self._get_operation_colors(metrics_df['operation'].unique())):
                print(f"Warning: More operations ({n_ops}) than colors ({len(self._get_operation_colors(metrics_df['operation'].unique()))})")
                self.operation_colors = sns.color_palette("husl", n_ops)
            
            print("Plotting duration distributions...")
            self._plot_duration_distributions(metrics_df, component, plot_dir)
            
            print("Plotting throughput variations...")
            self._plot_throughput_variations(metrics_df, component, plot_dir)
            
            print("Plotting memory usage...")
            self._plot_memory_usage(metrics_df, component, plot_dir)
            
            print("Plotting correlation heatmap...")
            self._plot_correlation_heatmap(metrics_df, component, plot_dir)
            
            print("Plotting metric trends...")
            self._plot_metric_trends(metrics_df, component, plot_dir)
            
            print(f"Completed plotting for {component}")
            
        except Exception as e:
            print(f"Error generating plots for {component}: {str(e)}")
            import traceback
            print(traceback.format_exc())

def analyze_benchmarks(component: str, output_dir: Optional[str] = None):
    """Analyze benchmarks for a component and optionally save results."""
    analyzer = BenchmarkAnalyzer()
    
    try:
        # Run analysis
        analysis = analyzer.analyze_metrics(component)
        
        # Generate report
        report = ["# Benchmark Analysis Report", ""]
        
        # System Info
        report.append("## System Information")
        for key, value in analysis['system_info'].items():
            report.append(f"- {key}: {value}")
        report.append("")
        
        # Metrics Summary
        report.append("## Performance Metrics")
        for operation, metrics in analysis['metrics_summary'].items():
            report.append(f"\n### {operation}")
            report.append(f"- Total executions: {metrics['count']}")
            
            report.append("\nDuration (ms):")
            for stat, value in metrics['duration_ms'].items():
                report.append(f"- {stat}: {value:.2f}")
            
            report.append("\nThroughput:")
            for stat, value in metrics['throughput'].items():
                report.append(f"- {stat}: {value:.2f}")
            
            if 'memory_mb' in metrics:
                report.append("\nMemory (MB):")
                for stat, value in metrics['memory_mb'].items():
                    report.append(f"- {stat}: {value:.2f}")
            
            # Add context metrics if available
            context_metrics = {k: v for k, v in metrics.items() if k.startswith('context_')}
            if context_metrics:
                report.append("\nContext Metrics:")
                for metric, values in context_metrics.items():
                    metric_name = metric.replace('context_', '')
                    report.append(f"\n{metric_name}:")
                    for stat, value in values.items():
                        report.append(f"- {stat}: {value:.2f}")
        
        # Error Summary
        if analysis['error_summary']:
            report.append("\n## Errors")
            for event_type, count in analysis['error_summary'].items():
                report.append(f"- {event_type}: {count}")
        
        # Save report and generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save text report
            report_file = output_path / f"{component}_analysis.md"
            with open(report_file, 'w') as f:
                f.write("\n".join(report))
            
            # Generate plots
            analyzer.plot_metrics(component, output_dir)
        
        return "\n".join(report)
    
    except Exception as e:
        import traceback
        return f"Error analyzing benchmarks: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    # Example usage
    print(analyze_benchmarks("whisper", "benchmark_reports")) 