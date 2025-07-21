import pickle
import argparse

def print_pickle_items(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {v}")
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                print(item)
        else:
            print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print items from a pickle file.")
    parser.add_argument("pkl_path", help="Path to the pickle file")
    args = parser.parse_args()
    print_pickle_items(args.pkl_path)