from utils.benchmark_analysis import analyze_benchmarks, BenchmarkAnalyzer
import os

# Create output directory
os.makedirs("benchmark_reports", exist_ok=True)

# Analyze each component
components = ["whisper", "vad", "audio", "diarization"]

analyzer = BenchmarkAnalyzer()

for component in components:
    print(f"\n=== {component.upper()} ANALYSIS ===\n")
    try:
        # Load data first to check what we're working with
        df = analyzer.load_component_data(component)
        print(f"Found {len(df)} records for {component}")
        
        # Print detailed information about the data
        print("\nData Types:")
        print(df['type'].value_counts())
        
        metrics_df = df[df['type'] == 'metric']
        if not metrics_df.empty:
            print("\nOperations found:")
            print(metrics_df['operation'].value_counts())
            print("\nMetrics columns:")
            print(metrics_df.columns.tolist())
        else:
            print("\nNo metrics found in the data!")
            
        # Run analysis
        result = analyze_benchmarks(component, "benchmark_reports")
        print("\nAnalysis Result:")
        print(result)
    except Exception as e:
        print(f"Error analyzing {component}: {str(e)}") 