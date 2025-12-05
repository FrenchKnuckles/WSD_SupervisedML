import os
import subprocess
import argparse
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run WSD project pipeline")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--lemmas", action="store_true")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    #parser.add_argument("--tune", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        args.data = args.lemmas = True
    #    args.tune = True
        args.models = True
        args.compare = args.features = args.graphs = True

    if args.data:
        run("python dataset_convert.py")

    if args.lemmas:
        run("python ambigious_lemma_extract.py")

    #if args.tune:
    #    run("python model_tuning.py")

    if args.models:
        run("python models/SVM.py")
        run("python models/Naive_Bayes.py")
        run("python models/decision_trees.py")
        run("python models/logistic_regression.py")

    if args.compare:
        run("python compare.py")

    if args.features:
        run("python feature_compare.py")

    if args.graphs:
        run("python graphical_compare.py")
        run("python interpretibility_compare.py")

if __name__ == "__main__":
    main()
