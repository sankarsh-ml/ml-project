"""Public classifier API. Training uses only the supplied training partition."""
import argparse
import pandas as pd
from model_selection import FEATURES, bls_features, feature_matrix, grouped_folds, train, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("features_csv")
    parser.add_argument("model_path")
    parser.add_argument("output_csv", nargs="?")
    args = parser.parse_args()
    data = pd.read_csv(args.features_csv, dtype={"tic_id": str})
    if args.mode == "train":
        train(data, args.model_path)
    elif args.output_csv:
        predict(data, args.model_path).to_csv(args.output_csv, index=False)
    else:
        parser.error("predict requires output_csv")
