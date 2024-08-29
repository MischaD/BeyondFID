import argparse
from beyondfid.feature_computation import compute_features 
from beyondfid.config import config

def main(args):
    compute_features(config, args.pathreal, args.pathsynth, args.output_path)
    #args.metrics, 

    #, args.results_filename


def get_args():
    parser = argparse.ArgumentParser(description="BeyondFID CLI")
    parser.add_argument("pathreal", type=str, help="data dir or csv with paths to data. Recursively looks through data dir")
    parser.add_argument("pathsynth", type=str, help="data dir or csv with paths to synthetic data. Recursively looks through data dir")
    parser.add_argument("--metrics", type=str, default="fid,is,recall,precision,coverage,density")
    parser.add_argument("--config", type=str, default="config.py", help="Configuration file. Defaults to config.py")
    parser.add_argument("--output_path", type=str, default="generative_metrics", help="Output path.")
    parser.add_argument("--results_filename", type=str, default="results.json", help="Name of file with results. Defaults to output_path/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.metrics = list(args.metrics.split(","))
    main(args)