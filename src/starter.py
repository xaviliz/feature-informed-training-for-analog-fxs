import argparse
import json
from train import training


def main(cfg: dict) -> None:

    for conditioning_type in cfg["conditioning_types"]:
        for feature in cfg["features"]:
            for loss in cfg["loss"]:
                for seq_len in cfg["seq_len"]:
                    training(
                        model=cfg["model"],
                        conditioning_type=conditioning_type,
                        dataset_name=cfg["dataset_name"],
                        data_dir=cfg["data_dir"],
                        embedding_dim=cfg["embedding_dim"],
                        feature=feature,
                        use_multiband=cfg["use_multiband"],
                        extractor=cfg["extractor"],
                        predict_feature=cfg["predict_feature"],
                        epochs=cfg["epochs"],
                        seq_len=seq_len,
                        hidden_size=cfg["hidden_size"],
                        input_size=cfg["input_size"],
                        order=cfg["order"],
                        lr=cfg["learning_rate"],
                        loss=loss,
                        from_scratch=cfg["from_scratch"],
                        lim_for_testing=cfg["lim_for_testing"],
                        extract_in_loading=cfg["extract_in_loading"],
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training a model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    print(f"Loading config file: {args.config}")
    with open(args.config, "r") as f:
        cfg = json.load(f)

    print(f"cfg: {cfg}")
    main(cfg)
