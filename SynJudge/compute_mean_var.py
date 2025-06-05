import json
import numpy as np
import argparse
import logging
from typing import List

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_jsonl(file_path: str) -> np.ndarray:
    """
    Reads a JSONL file and extracts the 'labels' field from each line.

    Args:
        file_path: The path to the JSONL file.

    Returns:
        A NumPy array containing the scores. Returns an empty array if the file is not found or is empty.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "labels" in item and isinstance(item["labels"], list):
                        data.append(item["labels"])
                    else:
                        logging.warning(f"Skipping line due to missing or invalid 'labels' field: {line.strip()}")
                except json.JSONDecodeError:
                    logging.warning(f"Skipping line due to JSON decoding error: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {file_path}")
        return np.array([])
        
    if not data:
        logging.warning("No valid data found in the input file.")
        return np.array([])
        
    return np.array(data)

def main(args):
    """
    Main function to calculate and save score statistics.
    """
    # Load data and convert it to a NumPy array
    logging.info(f"Reading scores from: {args.input_file}")
    scores = read_jsonl(args.input_file)

    if scores.size == 0:
        logging.error("No scores were loaded. Cannot calculate statistics. Exiting.")
        return

    # Compute mean and variance
    logging.info("Calculating mean and variance...")
    score_avg = scores.mean(axis=0).tolist()
    variance = scores.var(axis=0).tolist()
    score_all = float(np.mean(score_avg))
    variance_all = float(np.mean(variance))

    # Construct the output result dictionary
    result = {
        "score_mean_per_dimension": score_avg,
        "variance_per_dimension": variance,
        "overall_mean_score": score_all,
        "overall_mean_variance": variance_all
    }

    # Save the result to the output file
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logging.info(f"The result has been saved to: {args.output_file}")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except IOError as e:
        logging.error(f"Failed to write to output file {args.output_file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate mean and variance from a JSONL file containing scores.")

    parser.add_argument('--input_file', type=str, default='model_score.jsonl',
                        help='Path to the input JSONL file. Each line should be a JSON object with a "labels" key.')
    parser.add_argument('--output_file', type=str, default='mean_var_score.jsonl',
                        help='Path to the output file where the results will be saved.')
    
    args = parser.parse_args()
    main(args)
