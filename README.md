# Hugging Face Model Benchmarking Suite

A comprehensive suite for downloading, testing, and benchmarking Hugging Face models with resource monitoring.

## Features

- **Automated Model Download**: Downloads the smallest/quantized versions of specified models
- **Comprehensive Benchmarking**: Tests models with various prompts and measures performance
- **Resource Monitoring**: Tracks CPU, GPU, and memory usage during inference
- **Detailed Reporting**: Generates comprehensive reports with performance comparisons
- **Kaggle Optimized**: Designed to run efficiently in Kaggle environments

## Models Tested

1. **Llama-3.2-1B** (~1.2GB) - Meta's efficient language model
2. **TinyLlama 1.1B** (~300MB) - Ultra-compact language model
3. **SmolVLM-Instruct** (~1.1GB Q4_K_M GGUF) - HuggingFace's vision-language model

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main_orchestrator.py
```

This will:
- Download all models (quantized versions preferred)
- Run benchmark tests with sample prompts
- Monitor resource usage
- Generate comprehensive reports

### 3. View Results

Results will be saved in the `./results/` directory:
- `benchmark_results.json` - Detailed benchmark data
- `summary_report.json` - High-level summary and comparisons
- `system_info.json` - System specifications
- `pipeline.log` - Execution logs

## Advanced Usage

### Custom Configuration

```bash
# Use specific device
python main_orchestrator.py --device cuda

# Force re-download models
python main_orchestrator.py --force-download

# Skip certain phases
python main_orchestrator.py --skip-download  # Use previously downloaded models
python main_orchestrator.py --skip-benchmarks  # Just generate reports

# Custom directories
python main_orchestrator.py --cache-dir ./my_models --output-dir ./my_results
```

### Individual Components

```bash
# Download models only
python model_downloader.py --cache-dir ./models

# Run benchmarks only (requires downloaded models)
python model_tester.py --models-dir ./models --output ./benchmarks.json

# Generate visualization report
python report_generator.py --input ./results/benchmark_results.json
```

### For Private Models

```bash
# Set Hugging Face token
python main_orchestrator.py --hf-token YOUR_HF_TOKEN
```

## Kaggle-Specific Setup

### In Kaggle Notebook

1. **Install Requirements**:
```python
!pip install -r requirements.txt
```

2. **Run Pipeline**:
```python
!python main_orchestrator.py --device cuda --log-level INFO
```

3. **Monitor Progress**:
```python
# Check logs
!tail -f results/pipeline.log

# Check system resources
from resource_monitor import quick_resource_check
print(quick_resource_check())
```

### Kaggle GPU Optimization

The pipeline automatically:
- Uses 4-bit quantization when GPU memory is limited
- Implements memory cleanup between model tests
- Prefers quantized model variants to save space
- Monitors GPU temperature and utilization

## Output Metrics

For each model, the suite measures:

- **Performance**: Response time, tokens per second
- **Memory**: Peak/average RAM and GPU memory usage
- **Efficiency**: GPU utilization, temperature monitoring
- **Quality**: Generated text samples for evaluation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use `--device cpu` for testing
   - The pipeline automatically uses quantization
   - Reduce batch sizes in model configurations

2. **Model Download Failures**:
   - Check internet connection
   - Verify Hugging Face token for private models
   - Use `--force-download` to retry

3. **Import Errors**:
   - Ensure all requirements are installed
   - Check Python version compatibility (3.8+)

### Debug Mode

```bash
python main_orchestrator.py --log-level DEBUG
```

## File Structure

```
.
├── main_orchestrator.py      # Main pipeline coordinator
├── model_downloader.py       # Downloads models from HF
├── model_tester.py          # Runs benchmark tests
├── resource_monitor.py      # System resource monitoring
├── report_generator.py      # Visualization and reporting
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Downloaded models (created)
│   ├── llama-3.2-1b/
│   ├── tinyllama/
│   ├── kosmos-2/
│   └── phi-3.5-vision/
└── results/               # Output files (created)
    ├── benchmark_results.json
    ├── summary_report.json
    ├── system_info.json
    └── pipeline.log
```

## Performance Expectations

Typical runtimes on Kaggle T4 GPU:
- Model downloads: 10-30 minutes (depending on sizes)
- Benchmarking: 5-15 minutes per model
- Report generation: 1-2 minutes

Memory requirements:
- System RAM: 8GB+ recommended
- GPU Memory: 4GB+ for quantized models
- Storage: 10-50GB for all models

## Contributing

To add new models:
1. Update `models_config` in `model_downloader.py`


2. Add model-specific loading logic in `model_tester.py`
3. Update test prompts if needed

## License

This project is open source. Model licenses vary by provider - check individual model cards on Hugging Face.