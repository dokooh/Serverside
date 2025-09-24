#!/usr/bin/env python3
"""
Main Orchestrator Script
Coordinates the entire model download, testing, and reporting process
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any

# Import our custom modules
from model_downloader import ModelDownloader
from model_tester import ModelTester
from resource_monitor import quick_resource_check

logger = logging.getLogger(__name__)

class ModelBenchmarkOrchestrator:
    """Orchestrates the complete model benchmarking pipeline"""
    
    def __init__(self, 
                 cache_dir: str = "./models",
                 output_dir: str = "./results",
                 device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # File paths
        self.download_results_file = self.cache_dir / "download_results.json"
        self.benchmark_results_file = self.output_dir / "benchmark_results.json"
        self.system_info_file = self.output_dir / "system_info.json"
        self.summary_report_file = self.output_dir / "summary_report.json"
        
        # Initialize components
        self.downloader = None
        self.tester = None
    
    def check_system_requirements(self) -> Dict:
        """Check system requirements and capabilities"""
        logger.info("Checking system requirements...")
        
        system_info = quick_resource_check()
        
        # Add Python and library versions
        import torch
        import transformers
        
        system_info.update({
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "recommended_device": "cuda" if torch.cuda.is_available() else "cpu"
        })
        
        # Save system info
        with open(self.system_info_file, 'w') as f:
            json.dump(system_info, f, indent=2)
        
        logger.info(f"System info saved to {self.system_info_file}")
        return system_info
    
    def download_models(self, force_download: bool = False, hf_token: str = None) -> Dict:
        """Download all required models"""
        logger.info("🚀 === PHASE 1: DOWNLOADING MODELS ===")
        logger.debug(f"🔧 Debug - Cache directory: {self.cache_dir}")
        logger.debug(f"🔧 Debug - Force download: {force_download}")
        logger.debug(f"🔧 Debug - HF token provided: {hf_token is not None}")
        logger.debug(f"🔧 Debug - Starting download phase...")
        
        # Initialize downloader
        if hf_token:
            logger.debug("🔑 Debug - Logging in to Hugging Face...")
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("✅ Logged in to Hugging Face")
            logger.debug(f"✅ Debug - HF authentication completed")
        
        logger.debug("🔧 Debug - Initializing model downloader...")
        self.downloader = ModelDownloader(cache_dir=str(self.cache_dir))
        logger.info(f"✅ Downloader initialized with {len(self.downloader.models_config)} models configured")
        logger.debug(f"🔧 Debug - Models to download: {list(self.downloader.models_config.keys())}")
        
        # Download models
        logger.info("📥 Starting model downloads...")
        logger.debug(f"🔧 Debug - Calling download_all_models with force_download={force_download}")
        download_results = self.downloader.download_all_models(force_download=force_download)
        logger.debug(f"🔧 Debug - Download process completed, processing results...")
        
        # Log results
        logger.info("📊 Download Summary:")
        total_size = 0
        successful_downloads = 0
        
        logger.debug(f"🔧 Debug - Processing {len(download_results)} download results...")
        for model_key, result in download_results.items():
            logger.debug(f"🔧 Debug - Processing result for {model_key}: {result}")
            if "error" in result:
                logger.error(f"  ❌ {model_key}: FAILED - {result['error']}")
                logger.debug(f"🔧 Debug - Error details for {model_key}: {result}")
            else:
                size = result['size_gb']
                total_size += size
                successful_downloads += 1
                logger.info(f"  ✅ {model_key}: SUCCESS - {result['repo_id']} ({size:.2f} GB)")
                logger.debug(f"🔧 Debug - {model_key} files: {len(result.get('files', []))} files")
        
        logger.info(f"Total: {successful_downloads}/{len(download_results)} models downloaded ({total_size:.2f} GB)")
        
        return download_results
    
    def run_benchmarks(self) -> List:
        """Run benchmarks on all downloaded models"""
        logger.info("🧪 === PHASE 2: RUNNING BENCHMARKS ===")
        logger.debug(f"🔧 Debug - Models directory: {self.cache_dir}")
        logger.debug(f"🔧 Debug - Target device: {self.device}")
        logger.debug(f"🔧 Debug - Starting benchmark phase...")
        
        # Check if download results exist
        logger.debug(f"🔍 Debug - Checking for download results at {self.download_results_file}")
        if not self.download_results_file.exists():
            logger.error(f"❌ Download results file not found: {self.download_results_file}")
            logger.debug(f"🔧 Debug - Expected file path: {self.download_results_file}")
            raise FileNotFoundError(f"Download results not found: {self.download_results_file}")
        
        logger.debug(f"✅ Debug - Download results found: {self.download_results_file}")
        
        # Initialize tester
        logger.debug("🔧 Debug - Initializing model tester...")
        logger.debug(f"🔧 Debug - Using models_dir={self.cache_dir}, device={self.device}")
        self.tester = ModelTester(models_dir=str(self.cache_dir), device=self.device)
        
        logger.info(f"✓ Tester initialized - Using device: {self.tester.device}")
        logger.debug(f"Test prompts available: {len(self.tester.test_prompts)} text, {len(self.tester.vision_prompts)} vision")
        
        # Run tests
        logger.info("🚀 Starting model benchmarks...")
        benchmark_results = self.tester.test_all_models(str(self.download_results_file))
        
        # Save results
        self.tester.save_test_results(benchmark_results, str(self.benchmark_results_file))
        
        # Log summary
        summary = self.tester.benchmark_runner.get_summary_stats()
        logger.info("Benchmark Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return benchmark_results
    
    def generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report"""
        logger.info("=== PHASE 3: GENERATING REPORT ===")
        
        # Load all data
        with open(self.download_results_file, 'r') as f:
            download_results = json.load(f)
        
        with open(self.benchmark_results_file, 'r') as f:
            benchmark_results = json.load(f)
        
        with open(self.system_info_file, 'r') as f:
            system_info = json.load(f)
        
        # Analyze results by model
        model_summaries = {}
        
        for model_key in download_results.keys():
            if "error" in download_results[model_key]:
                model_summaries[model_key] = {
                    "status": "download_failed",
                    "error": download_results[model_key]["error"]
                }
                continue
            
            # Get benchmark results for this model
            model_benchmarks = [r for r in benchmark_results if r["model_name"].startswith(model_key)]
            
            if not model_benchmarks:
                model_summaries[model_key] = {
                    "status": "no_benchmarks",
                    "model_info": download_results[model_key]
                }
                continue
            
            # Calculate averages for successful benchmarks
            successful_benchmarks = [r for r in model_benchmarks if r["error"] is None]
            
            if not successful_benchmarks:
                model_summaries[model_key] = {
                    "status": "all_benchmarks_failed",
                    "model_info": download_results[model_key],
                    "benchmark_count": len(model_benchmarks),
                    "errors": [r["error"] for r in model_benchmarks if r["error"]]
                }
                continue
            
            # Calculate metrics
            avg_response_time = sum(r["response_time_seconds"] for r in successful_benchmarks) / len(successful_benchmarks)
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_benchmarks) / len(successful_benchmarks)
            avg_peak_memory = sum(r["peak_memory_gb"] for r in successful_benchmarks) / len(successful_benchmarks)
            avg_peak_gpu_memory = sum(r["peak_gpu_memory_gb"] for r in successful_benchmarks) / len(successful_benchmarks)
            avg_gpu_utilization = sum(r["avg_gpu_utilization"] for r in successful_benchmarks) / len(successful_benchmarks)
            
            model_summaries[model_key] = {
                "status": "success",
                "model_info": download_results[model_key],
                "benchmark_stats": {
                    "total_benchmarks": len(model_benchmarks),
                    "successful_benchmarks": len(successful_benchmarks),
                    "avg_response_time_seconds": avg_response_time,
                    "avg_tokens_per_second": avg_tokens_per_sec,
                    "avg_peak_memory_gb": avg_peak_memory,
                    "avg_peak_gpu_memory_gb": avg_peak_gpu_memory,
                    "avg_gpu_utilization_percent": avg_gpu_utilization
                }
            }
        
        # Create comprehensive report
        summary_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": system_info,
            "overall_stats": {
                "models_tested": len(download_results),
                "successful_downloads": len([m for m in download_results.values() if "error" not in m]),
                "models_with_successful_benchmarks": len([m for m in model_summaries.values() if m["status"] == "success"]),
                "total_benchmark_runs": len(benchmark_results),
                "successful_benchmark_runs": len([r for r in benchmark_results if r["error"] is None])
            },
            "model_summaries": model_summaries,
            "performance_comparison": self._create_performance_comparison(model_summaries)
        }
        
        # Save report
        with open(self.summary_report_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"Summary report saved to {self.summary_report_file}")
        return summary_report
    
    def _create_performance_comparison(self, model_summaries: Dict) -> Dict:
        """Create performance comparison between models"""
        successful_models = {k: v for k, v in model_summaries.items() if v["status"] == "success"}
        
        if not successful_models:
            return {"error": "No successful models to compare"}
        
        # Find best and worst performers
        metrics = ["avg_response_time_seconds", "avg_tokens_per_second", "avg_peak_memory_gb", "avg_peak_gpu_memory_gb"]
        comparison = {}
        
        for metric in metrics:
            values = [(k, v["benchmark_stats"][metric]) for k, v in successful_models.items()]
            
            if metric in ["avg_response_time_seconds", "avg_peak_memory_gb", "avg_peak_gpu_memory_gb"]:
                # Lower is better
                best = min(values, key=lambda x: x[1])
                worst = max(values, key=lambda x: x[1])
            else:
                # Higher is better
                best = max(values, key=lambda x: x[1])
                worst = min(values, key=lambda x: x[1])
            
            comparison[metric] = {
                "best": {"model": best[0], "value": best[1]},
                "worst": {"model": worst[0], "value": worst[1]},
                "all_values": dict(values)
            }
        
        return comparison
    
    def run_full_pipeline(self, 
                         force_download: bool = False,
                         hf_token: str = None,
                         skip_download: bool = False,
                         skip_benchmarks: bool = False) -> Dict:
        """Run the complete pipeline"""
        logger.info("🚀 Starting full model benchmarking pipeline...")
        logger.debug(f"Pipeline parameters:")
        logger.debug(f"  - force_download: {force_download}")
        logger.debug(f"  - skip_download: {skip_download}")
        logger.debug(f"  - skip_benchmarks: {skip_benchmarks}")
        logger.debug(f"  - device: {self.device}")
        
        # Check system
        logger.info("🔍 Checking system requirements...")
        system_info = self.check_system_requirements()
        logger.info(f"✓ System check complete - Running on: {system_info['recommended_device']}")
        logger.debug(f"System info: {system_info}")
        
        try:
            # Phase 1: Download models
            if not skip_download:
                logger.info("📥 Phase 1: Model Download")
                download_results = self.download_models(force_download, hf_token)
                logger.info("✅ Phase 1 completed successfully")
            else:
                logger.info("⏭️ Skipping download phase")
                if not self.download_results_file.exists():
                    logger.error("No download results found and download skipped")
                    raise FileNotFoundError("No download results found and download skipped")
                logger.debug("✓ Found existing download results")
            
            # Phase 2: Run benchmarks
            if not skip_benchmarks:
                logger.info("🧪 Phase 2: Model Benchmarking")
                benchmark_results = self.run_benchmarks()
                logger.info("✅ Phase 2 completed successfully")
            else:
                logger.info("⏭️ Skipping benchmark phase")
                if not self.benchmark_results_file.exists():
                    logger.error("No benchmark results found and benchmarks skipped")
                    raise FileNotFoundError("No benchmark results found and benchmarks skipped")
                logger.debug("✓ Found existing benchmark results")
            
            # Phase 3: Generate report
            logger.info("📊 Phase 3: Report Generation")
            summary_report = self.generate_summary_report()
            logger.info("✅ Phase 3 completed successfully")
            
            logger.info("🎉 === PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"📁 Results saved in: {self.output_dir}")
            logger.debug(f"Generated files:")
            logger.debug(f"  - {self.benchmark_results_file}")
            logger.debug(f"  - {self.summary_report_file}")
            logger.debug(f"  - {self.system_info_file}")
            
            return summary_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Hugging Face Model Benchmark Pipeline")
    
    # Directory options
    parser.add_argument("--cache-dir", default="./models", help="Directory to store downloaded models")
    parser.add_argument("--output-dir", default="./results", help="Directory to store results")
    
    # Execution options
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of models")
    parser.add_argument("--hf-token", help="Hugging Face token for private models")
    
    # Skip options
    parser.add_argument("--skip-download", action="store_true", help="Skip model download phase")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark phase")
    
    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Create output directory first (needed for log file)
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.output_dir}/pipeline.log")
        ]
    )
    
    # Initialize orchestrator
    orchestrator = ModelBenchmarkOrchestrator(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    try:
        # Run pipeline
        summary_report = orchestrator.run_full_pipeline(
            force_download=args.force_download,
            hf_token=args.hf_token,
            skip_download=args.skip_download,
            skip_benchmarks=args.skip_benchmarks
        )
        
        # Print quick summary
        print("\n" + "="*60)
        print("BENCHMARK PIPELINE COMPLETED")
        print("="*60)
        print(f"Results saved in: {args.output_dir}")
        print(f"Models tested: {summary_report['overall_stats']['models_tested']}")
        print(f"Successful benchmarks: {summary_report['overall_stats']['successful_benchmark_runs']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()