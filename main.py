import argparse
from src.pipeline import run_pipeline

data_path = "./data/Social_Network_Ads.csv"     # default data path

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="command-line interface to train different models directly from command line.")
    parser.add_argument("--path", "-p", default=data_path, help="file path or url for file")
    parser.add_argument("--auto")
    subparser = parser.add_subparsers(dest="subcommand")

    parser_dt = subparser.add_parser("decisiontree")
    parser_dt.add_argument("--criterion", "-c", default="gini")
    parser_dt.add_argument("--max_depth", "-d", type=int, nargs="?", const=None)
    parser_dt.add_argument("--min_samples_split", "-s", type=int, default=2)

    parser_rf = subparser.add_parser("randomforest")
    parser_rf.add_argument("--n_estimators", "-n", type=int, default=100)
    parser_rf.add_argument("--criterion", "-c", default="gini")
    parser_rf.add_argument("--max_depth", "-d", type=int,nargs="?", const=None)
    parser_rf.add_argument("--min_samples_split", "-s", type=int, default=2)
    parser_rf.add_argument("--max_features", "-m", type=int, nargs="?", const=None)

    parser_xgb = subparser.add_parser("xgboost")
    parser_xgb.add_argument("--learning_rate", "l", type=float, default=0.3)
    parser_xgb.add_argument("--max_depth", "-d", type=int, default=6)
    parser_xgb.add_argument("--n_estimators", "-n", type=int, default=100)


    args = parser.parse_args()

    if args.auto:
        model_name = args.auto
        params = "auto"
    
    elif args.subcommand=="decisiontree":
        model_name = "DecisionTree"
        params = {
            "criterion":[args.criterion],
            "max_depth":[args.max_depth],
            "min_samples_split":[args.min_samples_split]
        }

    elif args.subcommand=="randomforest":
        model_name = "RandomForest"
        params = {
            "n_estimators":[args.n_estimators],
            "criterion":[args.criterion],
            "max_depth":[args.max_depth],
            "min_samples_split":[args.min_samples_split],
            "max_features":[args.max_features]
        }

    elif args.subcommand=="xgboost":
        model_name = "XGBoost"
        params = {
            "learning_rate":[args.learning_rate],
            "n_estimators":[args.n_estimators],
            "max_depth":[args.max_depth]
        }

    try:
        run_pipeline(data_uri=args.path, model_name=args.model)
    except Exception as e:
        raise e