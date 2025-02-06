from utils.benchmark_analysis import analyze_benchmarks
import os

# Create output directory
os.makedirs("benchmark_reports", exist_ok=True)

# Analyze each component
components = ["whisper", "vad", "audio"]

for component in components:
    print(f"\n=== {component.upper()} ANALYSIS ===\n")
    try:
        result = analyze_benchmarks(component, "benchmark_reports")
        print(result)
    except Exception as e:
        print(f"Error analyzing {component}: {str(e)}") 