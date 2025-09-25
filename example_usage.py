#!/usr/bin/env python3
"""
Usage Example - Simple script showing how to use the benchmark suite
"""

import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_quick_benchmark():
    """Run a quick benchmark example"""
    logger.info("=== QUICK BENCHMARK EXAMPLE ===")
    
    # Import our modules
    from main_orchestrator import ModelBenchmarkOrchestrator
    
    # Initialize orchestrator
    orchestrator = ModelBenchmarkOrchestrator(
        cache_dir="./models",
        output_dir="./results",
        device="auto"  # Will use GPU if available, otherwise CPU
    )
    
    try:
        # Run the complete pipeline
        logger.info("Starting benchmark pipeline...")
        summary_report = orchestrator.run_full_pipeline()
        
        # Print results
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED!")
        print("="*60)
        
        print(f"Models tested: {summary_report['overall_stats']['models_tested']}")
        print(f"Successful downloads: {summary_report['overall_stats']['successful_downloads']}")
        print(f"Successful benchmarks: {summary_report['overall_stats']['successful_benchmark_runs']}")
        
        print("\nTop performing models:")
        comparison = summary_report.get('performance_comparison', {})
        if 'avg_tokens_per_second' in comparison:
            best_speed = comparison['avg_tokens_per_second']['best']
            print(f"  Fastest: {best_speed['model']} ({best_speed['value']:.1f} tokens/sec)")
        
        if 'avg_response_time_seconds' in comparison:
            best_time = comparison['avg_response_time_seconds']['best']
            print(f"  Quickest response: {best_time['model']} ({best_time['value']:.2f} seconds)")
        
        print(f"\nDetailed results saved in: ./results/")
        print("  - summary_report.json: Complete analysis")
        print("  - benchmark_results.json: Raw benchmark data")
        print("  - system_info.json: System specifications")
        
        return True
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False

def run_individual_components():
    """Example of running individual components"""
    logger.info("=== INDIVIDUAL COMPONENTS EXAMPLE ===")
    
    # 1. Download models only
    logger.info("Step 1: Download models")
    from model_downloader import ModelDownloader
    
    downloader = ModelDownloader(models_dir="./models")
    download_results = downloader.download_all_models()
    
    print(f"Downloaded {len(download_results)} models")
    
    # 2. Test models
    logger.info("Step 2: Run benchmarks")
    from model_tester import ModelTester
    
    tester = ModelTester(models_dir="./models")
    results = tester.test_all_models("./models/download_results.json")
    tester.save_test_results(results, "./benchmark_results.json")
    
    print(f"Completed {len(results)} benchmark tests")
    
    # 3. Generate reports
    logger.info("Step 3: Generate reports")
    from report_generator import ModelPerformanceReporter
    
    reporter = ModelPerformanceReporter(output_dir="./reports")
    html_report = reporter.generate_html_report("./benchmark_results.json")
    
    print(f"Report generated: {html_report}")

def check_system_requirements():
    """Check if system meets requirements"""
    logger.info("=== SYSTEM REQUIREMENTS CHECK ===")
    
    from resource_monitor import quick_resource_check
    import torch
    
    # System check
    resources = quick_resource_check()
    
    print("System Information:")
    print(f"  CPU: {resources['cpu_percent']:.1f}% usage")
    print(f"  RAM: {resources['memory_used_gb']:.1f}/{resources['memory_total_gb']:.1f} GB")
    print(f"  GPU Available: {resources['gpu_available']}")
    
    if resources['gpu_available']:
        print(f"  GPU Count: {resources['gpu_count']}")
        print(f"  GPU Memory: {resources['gpu_memory_used_gb']:.1f}/{resources['gpu_memory_total_gb']:.1f} GB")
    
    # Recommendations
    print("\nRecommendations:")
    if resources['memory_total_gb'] < 8:
        print("  ⚠️  Warning: Less than 8GB RAM may cause issues with larger models")
    else:
        print("  ✓ RAM looks sufficient")
    
    if not resources['gpu_available']:
        print("  ⚠️  No GPU detected - will use CPU (slower)")
    else:
        print("  ✓ GPU available for acceleration")
    
    return resources

def main():
    """Main function with menu"""
    print("Hugging Face Model Benchmark Suite")
    print("==================================")
    
    while True:
        print("\nOptions:")
        print("1. Check system requirements")
        print("2. Run quick benchmark (full pipeline)")
        print("3. Run individual components")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            check_system_requirements()
        elif choice == "2":
            success = run_quick_benchmark()
            if success:
                print("\n✓ Benchmark completed successfully!")
            else:
                print("\n❌ Benchmark failed - check logs for details")
        elif choice == "3":
            run_individual_components()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()