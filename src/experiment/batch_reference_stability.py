#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch MOF Stability Prediction Script

This script uses subprocess to run reference stability prediction scripts on multiple CIF files 
and outputs the results in JSON format. It processes both thermal/solvent stability using 
reference_ts_stability.py and water/acid/base/boiling stability using reference_ws_stability.py.

Different Python environments can be specified for each stability script, as they may have
different dependencies.

Usage:
    python batch_reference_stability.py --input_dir /path/to/cifs --output_dir /path/to/output \
        --python_executable_ts /path/to/python/for/ts \
        --python_executable_ws /path/to/python/for/ws
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed


def find_cif_files(directory: str) -> List[str]:
    """
    Find all CIF files in the specified directory.
    
    Args:
        directory: The directory to search for CIF files.
        
    Returns:
        A list of paths to CIF files.
    """
    cif_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.cif'):
                cif_files.append(os.path.join(root, file))
    
    return cif_files


def run_predict_stability(cif_file: str, working_dir: str, python_executable_ts: Optional[str] = None,
                          python_executable_ws: Optional[str] = None) -> Dict[str, Union[str, float]]:
    """
    Run stability prediction scripts on a single CIF file.

    Args:
        cif_file: Path to the CIF file.
        working_dir: Directory to store intermediate files.
        python_executable_ts: Path to the Python executable for thermal/solvent stability prediction.
        python_executable_ws: Path to the Python executable for water/acid/base/boiling stability prediction.
        
    Returns:
        A dictionary containing the prediction results.
    """
    # Get the path to the prediction scripts
    script_path_ts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_ts_stability.py")
    script_path_ws = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_ws_stability.py")
    
    # Create a temporary directory for this prediction
    mof_name = os.path.splitext(os.path.basename(cif_file))[0]
    mof_working_dir = os.path.join(working_dir, f"tmp_{mof_name}")
    os.makedirs(mof_working_dir, exist_ok=True)
    
    # Use the specified Python executable or default to the system's Python
    exe_ts = python_executable_ts or sys.executable
    exe_ws = python_executable_ws or sys.executable
    
    # Command to run thermal and solvent stability prediction
    cmd_ts = [
        exe_ts,  # Python executable for thermal/solvent stability
        script_path_ts,
        cif_file,
        "--output_dir", mof_working_dir,
        "--keep_files",  # Keep intermediate files for debugging if needed
        "--json_output"  # Use JSON output format for easier parsing
    ]

    # Command to run water and acid stability prediction
    cmd_ws = [
        exe_ws,  # Python executable for water/acid/base/boiling stability
        script_path_ws,
        cif_file,
        "--output_dir", mof_working_dir,
        "--keep_files",  # Keep intermediate files for debugging if needed
        "--json_output"  # Use JSON output format for easier parsing
    ]
    
    print(f"Processing {mof_name}...")
    print(f"Command (Thermal and solvent stability): {' '.join(cmd_ts)}")
    print(f"Command (Water and acid stability): {' '.join(cmd_ws)}")

    start_time = time.time()
    results = {
        "MofName": mof_name
    }
    
    # Run the thermal/solvent stability script and capture output
    try:
        output_ts = subprocess.check_output(cmd_ts, stderr=subprocess.STDOUT, text=True)
        
        # Parse JSON output
        json_start = output_ts.find("--- JSON OUTPUT ---")
        if json_start >= 0:
            json_text = output_ts[json_start + len("--- JSON OUTPUT ---"):].strip()
            json_data = json.loads(json_text)
            
            # Record execution time
            results["execution_time_ts"] = json_data.get("execution_time", 0)
            
            predictions = json_data.get("predictions", {})
            
            # Extract thermal stability (removing the degree symbol if present)
            thermal = predictions.get("thermal_stability", "")
            print(f"Thermal stability prediction: {thermal}")
            if isinstance(thermal, str) and "°" in thermal:
                results["ts_label"] = float(thermal.replace("°C", ""))
            elif isinstance(thermal, (int, float)):
                results["ts_label"] = float(thermal)
            
            # Extract solvent stability value
            if "solvent_stability" in predictions:
                print(f"Solvent stability prediction: {predictions['solvent_stability']}")
                results["ss_label"] = float(predictions["solvent_stability"])
        else:
            # Fallback to regex parsing if JSON output is not found
            # Extract thermal stability (e.g., "350.0°C")
            thermal_match = re.search(r"Thermal Stability:\s+([0-9.]+)°C", output_ts)
            if thermal_match:
                print(f"Thermal stability prediction: {thermal_match.group(1)}")
                results["ts_label"] = float(thermal_match.group(1))
            
            # Extract solvent stability probability
            solvent_match = re.search(r"Solvent Stability:\s+([0-9.]+)", output_ts)
            if solvent_match:
                print(f"Solvent stability prediction: {solvent_match.group(1)}")
                results["ss_label"] = float(solvent_match.group(1))

        print(f"Thermal/solvent stability prediction completed for {mof_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing thermal/solvent stability for {mof_name}: {e}")
        if hasattr(e, 'output') and e.output:
            print(f"Output: {e.output}")
        results["error_ts"] = str(e)
    except Exception as e:
        print(f"Unexpected error processing thermal/solvent stability for {mof_name}: {e}")
        import traceback
        traceback.print_exc()
        results["error_ts"] = str(e)
    
    # Run the water/acid/base/boiling stability script and capture output
    try:
        output_ws = subprocess.check_output(cmd_ws, stderr=subprocess.STDOUT, text=True)
        
        # Parse JSON output
        json_start = output_ws.find("--- JSON OUTPUT ---")
        if json_start >= 0:
            json_text = output_ws[json_start + len("--- JSON OUTPUT ---"):].strip()
            json_data = json.loads(json_text)
            
            # Record execution time
            results["execution_time_ws"] = json_data.get("execution_time", 0)
            
            predictions = json_data.get("predictions", {})
            
            # Extract water and acid stability values
            if "water_stability" in predictions:
                print(f"Water stability prediction: {predictions['water_stability']}")
                results["water_label"] = float(predictions["water_stability"])

            if "water4_stability" in predictions:
                print(f"Water4 stability prediction: {predictions['water4_stability']}")
                results["water4_label"] = predictions["water4_stability"]
                
            if "acid_stability" in predictions:
                print(f"Acid stability prediction: {predictions['acid_stability']}")
                results["acid_label"] = float(predictions["acid_stability"])

            if "base_stability" in predictions:
                print(f"Base stability prediction: {predictions['base_stability']}")
                results["base_label"] = float(predictions["base_stability"])

            if "boiling_stability" in predictions:
                print(f"Boiling stability prediction: {predictions['boiling_stability']}")
                results["boiling_label"] = float(predictions["boiling_stability"])

        else:
            # Fallback to regex parsing if JSON output is not found
            # Extract water stability probability
            water_match = re.search(r"Water Stability:\s+([0-9.]+)", output_ws)
            if water_match:
                print(f"Water stability prediction: {water_match.group(1)}")
                results["water_label"] = float(water_match.group(1))

            # Extract water4 stability probability
            water4_match = re.search(r"Water4 Stability:\s+([\[,0-9.\]\s]+)", output_ws)
            if water4_match:
                print(f"Water4 stability prediction: {water4_match.group(1)}")
                results["water4_label"] = eval(water4_match.group(1))

            # Extract acid stability probability
            acid_match = re.search(r"Acid Stability:\s+([0-9.]+)", output_ws)
            if acid_match:
                print(f"Acid stability prediction: {acid_match.group(1)}")
                results["acid_label"] = float(acid_match.group(1))

            # Extract base stability probability
            base_match = re.search(r"Base Stability:\s+([0-9.]+)", output_ws)
            if base_match:
                print(f"Base stability prediction: {base_match.group(1)}")
                results["base_label"] = float(base_match.group(1))
            # Extract boiling stability probability
            boiling_match = re.search(r"Boiling Stability:\s+([0-9.]+)", output_ws)
            if boiling_match:
                print(f"Boiling stability prediction: {boiling_match.group(1)}")
                results["boiling_label"] = float(boiling_match.group(1))

        print(f"Water/acid/base/boiling stability prediction completed for {mof_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing water/acid/base/boiling stability for {mof_name}: {e}")
        if hasattr(e, 'output') and e.output:
            print(f"Output: {e.output}")
        results["error_ws"] = str(e)
    except Exception as e:
        print(f"Unexpected error processing water/acid/base/boiling stability for {mof_name}: {e}")
        import traceback
        traceback.print_exc()
        results["error_ws"] = str(e)
    
    end_time = time.time()
    results["total_execution_time"] = end_time - start_time
    print(f"Completed {mof_name} in {end_time - start_time:.2f} seconds")
    
    return results


def save_results(results: List[Dict[str, Union[str, float]]], output_dir: str) -> None:
    """
    Save results to json files.
    
    Args:
        results: List of prediction results.
        output_dir: Directory to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw JSON results for reference
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def process_batch(cif_files: List[str], working_dir: str, max_workers: int = 1, 
                 python_executable_ts: Optional[str] = None,
                 python_executable_ws: Optional[str] = None) -> List[Dict[str, Union[str, float]]]:
    """
    Process a batch of CIF files in parallel using ThreadPoolExecutor.
    
    Args:
        cif_files: List of CIF files to process.
        working_dir: Directory to store intermediate files.
        max_workers: Maximum number of workers to use for parallel processing.
        python_executable_ts: Path to the Python executable for thermal/solvent stability.
        python_executable_ws: Path to the Python executable for water/acid/base/boiling stability.
        
    Returns:
        List of prediction results for each CIF file.
    """
    results = []
    
    print(f"Processing {len(cif_files)} CIF files with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_file = {
            executor.submit(
                run_predict_stability, 
                cif_file, 
                working_dir, 
                python_executable_ts,
                python_executable_ws
            ): cif_file
            for cif_file in cif_files
        }
        
        # Process as they complete
        for i, future in enumerate(as_completed(future_to_file)):
            cif_file = future_to_file[future]
            mof_name = os.path.splitext(os.path.basename(cif_file))[0]
            
            try:
                result = future.result()
                results.append(result)
                print(f"[{i+1}/{len(cif_files)}] Completed {mof_name}")
            except Exception as exc:
                print(f"[{i+1}/{len(cif_files)}] {mof_name} generated an exception: {exc}")
                results.append({
                    "MofName": mof_name,
                    "error": str(exc)
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict MOF stability for multiple CIF files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing CIF files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output CSV files.")
    parser.add_argument("--working_dir", default=None, 
                       help="Directory for intermediate files (default: output_dir/working).")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit the number of CIF files to process (for testing).")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of parallel workers to use. Default: 1 (sequential).")
    parser.add_argument("--python_executable_ts", type=str, default="/opt/share/miniconda3/envs/MOFSimplify/bin/python",
                       help="Path to the Python executable for thermal/solvent stability prediction.")
    parser.add_argument("--python_executable_ws", type=str, default="/opt/share/miniconda3/envs/test/bin/python",
                       help="Path to the Python executable for water/acid/base/boiling stability prediction.")
    
    args = parser.parse_args()
    
    # Set working directory
    working_dir = args.working_dir or os.path.join(args.output_dir, "working")
    os.makedirs(working_dir, exist_ok=True)
    
    # Find all CIF files
    cif_files = find_cif_files(args.input_dir)
    print(f"Found {len(cif_files)} CIF files")
    
    if args.limit and args.limit > 0:
        cif_files = cif_files[:args.limit]
        print(f"Limited to processing {args.limit} CIF files")
    
    # Process CIF files in parallel or sequentially
    total_start_time = time.time()
    
    if args.workers > 1:
        results = process_batch(
            cif_files, 
            working_dir, 
            args.workers, 
            args.python_executable_ts,
            args.python_executable_ws
        )
    else:
        # Sequential processing for better error handling and debugging
        results = []
        for i, cif_file in enumerate(cif_files):
            print(f"[{i+1}/{len(cif_files)}] Processing {os.path.basename(cif_file)}")
            result = run_predict_stability(
                cif_file, 
                working_dir, 
                args.python_executable_ts,
                args.python_executable_ws
            )
            results.append(result)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds for {len(cif_files)} files")
    print(f"Average time per file: {total_time / len(cif_files):.2f} seconds")

    # Save results to JSON files
    save_results(results, args.output_dir)
    
    # Print statistics
    success_count = sum(1 for r in results if "error_ts" not in r and "error_ws" not in r)
    partial_success_count = sum(1 for r in results if ("error_ts" in r) != ("error_ws" in r))
    
    print(f"\nProcessing statistics:")
    print(f"  Total files:       {len(cif_files)}")
    print(f"  Fully successful:  {success_count} ({success_count / len(cif_files) * 100:.1f}%)")
    print(f"  Partial success:   {partial_success_count} ({partial_success_count / len(cif_files) * 100:.1f}%)")
    print(f"  Failed:            {len(cif_files) - success_count - partial_success_count} "
          f"({(len(cif_files) - success_count - partial_success_count) / len(cif_files) * 100:.1f}%)")


if __name__ == "__main__":
    main()
