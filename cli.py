
import argparse
from .train import train_from_csv, evaluate_from_csv

def train_cli():
    p = argparse.ArgumentParser(description="Train Music Mood Classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    train_from_csv(args.csv, args.out)

def eval_cli():
    p = argparse.ArgumentParser(description="Evaluate Music Mood Classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    evaluate_from_csv(args.csv, args.model)
