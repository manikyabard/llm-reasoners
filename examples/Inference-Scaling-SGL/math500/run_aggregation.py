#!/usr/bin/env python3

import json
import os
import time
import logging

def aggregate_results(output_path):
    """Aggregate results from chunk files and print summary."""
    total_start = time.time()
    
    final_output = {
        "metadata": {
            "total_time": time.time() - total_start,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(total_start)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "chunks_processed": 0,
            "success_rate": None
        },
        "results": []
    }

    # Load and merge chunk files
    chunk_files = [f for f in os.listdir() if f.startswith(f"{output_path}.chunk_")]
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "r") as f:
                chunk_data = json.load(f)
                final_output["results"].extend(chunk_data.get("results", []))
                final_output["metadata"]["chunks_processed"] += 1
            os.remove(chunk_file)
        except Exception as e:
            logging.error(f"Error processing {chunk_file}: {str(e)}")

    # Calculate statistics
    if final_output["results"]:
        processed = [r for r in final_output["results"] if r["status"] in ("completed", "failed")]
        success = [r for r in final_output["results"] if r["status"] == "completed"]
        times = [r["processing_time"] for r in processed if r["processing_time"] is not None]
        
        final_output["metadata"].update({
            "total_examples": len(processed),
            "success_count": len(success),
            "success_rate": len(success) / len(processed) if processed else 0,
            "time_stats": {
                "total": sum(times),
                "average": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0
            }
        })

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4, sort_keys=True)

    # Print summary
    try:
        logging.info("\nFinal Statistics:")
        logging.info(f"Total time: {final_output['metadata']['total_time']:.2f}s")
        logging.info(f"Processed examples: {final_output['metadata']['total_examples']}")
        logging.info(f"Success rate: {final_output['metadata']['success_rate']:.1%}")
        logging.info(f"Average processing time: {final_output['metadata']['time_stats']['average']:.2f}s")
    except Exception as e:
        logging.error(f"Error printing summary: {str(e)}")

if __name__ == "__main__":
    output_path = "math500_feb18_3.json"
    aggregate_results(output_path) 