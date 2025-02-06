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
            
        # Skip warmup period
        df = df.iloc[self.warmup_samples:]
        
        # Remove extreme outliers (beyond 99th percentile) for better visualization
        for col in ['duration_ms', 'throughput', 'memory_mb']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=q99)
        
        return df
    
    def _plot_duration_distributions(self, df, component, plot_dir):
        """Generate duration distribution plots."""
        try:
            plt.figure(figsize=(15, 8))
            
            # 1. Enhanced Violin Plot
            plt.subplot(2, 2, 1)
            if len(df['operation'].unique()) == 1:
                # Single operation - use primary color
                sns.violinplot(data=df, x='operation', y='duration_ms',
                             color='#e34a33', width=0.8, inner='box',
                             density_norm='width')
            else:
                # Multiple operations - use color scheme
                df_melted = df.copy()
                df_melted['hue'] = df_melted['operation']
                sns.violinplot(data=df_melted, x='operation', y='duration_ms',
                             hue='hue', legend=False,
                             palette=dict(zip(df['operation'].unique(),
                                            self._get_operation_colors(df['operation'].unique()))),
                             width=0.8, inner='box', density_norm='width')
            plt.title('Duration Distribution (ms)')
            plt.xticks(rotation=45)
            
            # 2. KDE Plot
            plt.subplot(2, 2, 2)
            operations = df['operation'].unique()
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                sns.kdeplot(data=op_data['duration_ms'],
                          color='#e34a33',
                          label=operations[0], fill=True, alpha=0.5)
            else:
                for operation in operations:
                    op_data = df[df['operation'] == operation]
                    if len(op_data) > 1:
                        sns.kdeplot(data=op_data['duration_ms'],
                                  color=self._get_operation_color(operation),
                                  label=operation, fill=True, alpha=0.3)
            plt.title('Duration Density (ms)')
            plt.legend()
            
            # 3. ECDF Plot
            plt.subplot(2, 2, 3)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                sns.ecdfplot(data=op_data, x='duration_ms',
                           color='#e34a33',
                           label=operations[0])
            else:
                for operation in operations:
                    op_data = df[df['operation'] == operation]
                    if len(op_data) > 1:
                        sns.ecdfplot(data=op_data, x='duration_ms',
                                   color=self._get_operation_color(operation),
                                   label=operation)
            plt.title('Duration Distribution (Cumulative) (ms)')
            plt.legend()
            
            # 4. Bar Plot with Error Bars
            plt.subplot(2, 2, 4)
            if len(operations) == 1:
                sns.barplot(data=df, x='operation', y='duration_ms',
                          color='#e34a33',
                          errorbar=('ci', 95))
            else:
                df_melted = df.copy()
                df_melted['hue'] = df_melted['operation']
                sns.barplot(data=df_melted, x='operation', y='duration_ms',
                          hue='hue', legend=False,
                          palette=dict(zip(df['operation'].unique(),
                                         self._get_operation_colors(df['operation'].unique()))),
                          errorbar=('ci', 95))
            plt.title('Mean Duration with 95% CI (ms)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            if plot_dir:
                plt.savefig(plot_dir / f'{component}_duration_dist.png',
                           dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting duration distributions: {str(e)}")
            plt.close()
    
    def _plot_throughput_variations(self, df, component, plot_dir):
        """Generate throughput visualization plots."""
        try:
            plt.figure(figsize=(15, 8))
            operations = df['operation'].unique()
            
            # 1. Line Plot with Rolling Mean
            plt.subplot(2, 2, 1)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                plt.plot(op_data.index, op_data['throughput'].rolling(window=20, min_periods=1).mean(),
                        color='#e34a33', label=operations[0], linewidth=2)
            else:
                colors = self._get_operation_colors(operations)
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if not op_data.empty:
                        plt.plot(op_data.index, op_data['throughput'].rolling(window=20, min_periods=1).mean(),
                                color=colors[i], label=operation, linewidth=2)
            plt.title('Throughput Over Time (samples/sec)')
            plt.ylabel('Throughput (samples/sec)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Distribution Plot
            plt.subplot(2, 2, 2)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                sns.kdeplot(data=op_data['throughput'],
                          color='#e34a33',
                          label=operations[0], fill=True, alpha=0.5)
            else:
                colors = self._get_operation_colors(operations)
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if len(op_data) > 1:
                        sns.kdeplot(data=op_data['throughput'],
                                  color=colors[i],
                                  label=operation, fill=True, alpha=0.3)
            plt.title('Throughput Distribution (samples/sec)')
            plt.xlabel('Throughput (samples/sec)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Box Plot
            plt.subplot(2, 2, 3)
            if len(operations) == 1:
                sns.boxplot(data=df, x='operation', y='throughput',
                          color='#e34a33')
            else:
                df_melted = df.copy()
                df_melted['hue'] = df_melted['operation']
                sns.boxplot(data=df_melted, x='operation', y='throughput',
                          hue='hue', legend=False,
                          palette=dict(zip(operations, colors)))
            plt.title('Throughput Range (samples/sec)')
            plt.ylabel('Throughput (samples/sec)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 4. Bar Plot
            plt.subplot(2, 2, 4)
            if len(operations) == 1:
                sns.barplot(data=df, x='operation', y='throughput',
                          color='#e34a33',
                          errorbar=('ci', 95))
            else:
                df_melted = df.copy()
                df_melted['hue'] = df_melted['operation']
                sns.barplot(data=df_melted, x='operation', y='throughput',
                          hue='hue', legend=False,
                          palette=dict(zip(operations, colors)),
                          errorbar=('ci', 95))
            plt.title('Mean Throughput with 95% CI (samples/sec)')
            plt.ylabel('Throughput (samples/sec)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if plot_dir:
                plt.savefig(plot_dir / f'{component}_throughput.png',
                           dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting throughput variations: {str(e)}")
            plt.close()
    
    def _plot_memory_usage(self, df, component, plot_dir):
        """Generate memory usage visualization plots."""
        if 'memory_mb' not in df.columns:
            return
            
        try:
            plt.figure(figsize=(15, 8))
            operations = df['operation'].unique()
            
            # 1. Line Plot with Rolling Mean
            plt.subplot(2, 2, 1)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                plt.plot(op_data.index, op_data['memory_mb'].rolling(window=20, min_periods=1).mean(),
                        color='#e34a33', label=operations[0], linewidth=2)
            else:
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if not op_data.empty:
                        plt.plot(op_data.index, op_data['memory_mb'].rolling(window=20, min_periods=1).mean(),
                                color=self._get_operation_color(operation), label=operation)
            plt.title('Memory Usage Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. KDE Plot
            plt.subplot(2, 2, 2)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                sns.kdeplot(data=op_data['memory_mb'],
                          color='#e34a33',
                          label=operations[0], fill=True, alpha=0.5)
            else:
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if len(op_data) > 1:
                        sns.kdeplot(data=op_data['memory_mb'],
                                  color=self._get_operation_color(operation),
                                  label=operation, fill=True, alpha=0.3)
            plt.title('Memory Usage Distribution')
            plt.legend()
            
            # 3. Stacked Area Plot
            plt.subplot(2, 2, 3)
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]['memory_mb'].rolling(window=20, min_periods=1).mean()
                plt.fill_between(df.index, op_data, color='#e34a33', alpha=0.7, label=operations[0])
            else:
                data = []
                for op in operations:
                    op_data = df[df['operation'] == op]['memory_mb'].rolling(window=20, min_periods=1).mean()
                    data.append(op_data)
                plt.stackplot(df.index, data, labels=operations, colors=self._get_operation_colors(operations), alpha=0.7)
            plt.title('Cumulative Memory Usage')
            plt.legend()
            
            # 4. Memory Usage Percentiles
            plt.subplot(2, 2, 4)
            percentiles = [25, 50, 75, 90, 95, 99]
            if len(operations) == 1:
                op_data = df[df['operation'] == operations[0]]
                if not op_data.empty:
                    perc_values = [np.percentile(op_data['memory_mb'], p) for p in percentiles]
                    plt.plot(percentiles, perc_values, 'o-',
                            color='#e34a33',
                            label=operations[0], linewidth=2, markersize=8)
            else:
                for i, operation in enumerate(operations):
                    op_data = df[df['operation'] == operation]
                    if not op_data.empty:
                        perc_values = [np.percentile(op_data['memory_mb'], p) for p in percentiles]
                        plt.plot(percentiles, perc_values, 'o-',
                                color=self._get_operation_color(operation),
                                label=operation, linewidth=2, markersize=8)
            plt.title('Memory Usage Percentiles')
            plt.xlabel('Percentile')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if plot_dir:
                plt.savefig(plot_dir / f'{component}_memory.png',
                           dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting memory usage: {str(e)}")
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