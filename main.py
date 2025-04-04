import argparse
from src.pipeline import run_pipeline

data_path = "./data/Social_Network_Ads.csv"     # default data path

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="command-line interface to train different models directly from command line.")
    parser.add_argument("--model", "-m", default="XGBoost")
    parser.add_argument("--path", "-p", default=data_path, help="file path or url for file")
    args = parser.parse_args()
    try:
        run_pipeline(data_uri=args.path, model_name=args.model)
    except Exception as e:
        raise e