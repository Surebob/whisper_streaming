# Benchmark Analysis Report

## System Information
- platform: Windows-10-10.0.22631-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- cpu_count: 32
- memory_total: 68387934208
- component: whisper

## Performance Metrics

### inference
- Total executions: 260

Duration (ms):
- mean: 345.12
- median: 347.86
- min: 120.42
- max: 765.94
- std: 105.56

Throughput:
- mean: 51903.87
- median: 45995.70
- min: 20889.30
- max: 132869.95

Memory (MB):
- mean: 1.69
- max: 143.43
- min: 0.00

Context Metrics:

operation_count:
- mean: 129.50
- max: 259.00
- min: 0.00

text_length:
- mean: 16.53
- max: 56.00
- min: 3.00

### process
- Total executions: 260

Duration (ms):
- mean: 347.31
- median: 349.12
- min: 121.02
- max: 780.50
- std: 106.42

Throughput:
- mean: 51590.94
- median: 45830.04
- min: 20499.73
- max: 132206.71

Memory (MB):
- mean: 1.69
- max: 143.43
- min: 0.00

Context Metrics:

is_speech:
- mean: 1.00
- max: 1.00
- min: 1.00

chunk_size:
- mean: 16000.00
- max: 16000.00
- min: 16000.00

operation_count:
- mean: 129.50
- max: 259.00
- min: 0.00

### finalize
- Total executions: 13

Duration (ms):
- mean: 0.61
- median: 0.52
- min: 0.12
- max: 1.76
- std: 0.50

Throughput:
- mean: 378971416.89
- median: 323848077.21
- min: 51052967.48
- max: 808664262.30

Memory (MB):
- mean: 0.48
- max: 1.83
- min: 0.00

Context Metrics:

final_text_length:
- mean: 261.08
- max: 1693.00
- min: 0.00

total_audio_samples:
- mean: 192000.00
- max: 480000.00
- min: 32000.00

silence_duration:
- mean: 2.32
- max: 2.65
- min: 2.05

is_valid_segment:
- mean: 0.85
- max: 1.00
- min: 0.00

words_per_second:
- mean: 3.09
- max: 11.33
- min: 0.80

operation_count:
- mean: 6.00
- max: 12.00
- min: 0.00