import sys
from pathlib import Path
from itertools import product

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from code.datasets import GSClassificationDataset

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Convert slides into images")  
    parser.add_argument("index_path", type=str, help="")
    parser.add_argument("pickle_split_path", type=str, help="")
    parser.add_argument("total_fold", type=int, help="")
    


for k_fold in range(args.total_folds):
    if True:
        try:
            GSClassificationDataset.make_dataset_index(
                pickle_split_path=args.pickle_split_path,
                k_fold=k_fold,
                index_path=args.index_path,
                ignore=[],
                total_folds=args.total_folds,
            )
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            print(f"[Error] Args: {args.pickle_split_path} {k_fold} {args.index_path} {args.total_folds}")
