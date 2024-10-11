import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model parameters
    parser.add_argument(
        "--method",
        type=str,
        default="salad",
        choices=[
            "netvlad",
            "sfrs",
            "cosplace",
            "convap",
            "mixvpr",
            "eigenplaces",
            "salad",
        ],
        help="_",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--num_epochs", type=int, default=5, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")

    parser.add_argument("--lr", type=float, default=0.00001, help="_")

    parser.add_argument("--train_on_southern_half", action="store_true", help="_")

    # Validation / test parameters
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (validating and testing)",
    )

    # Resume parameters
    parser.add_argument(
        "--resume_model",
        type=str,
        default=None,
        help="path to model to resume, e.g. logs/.../best_model.pth",
    )

    # Other parameters
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument(
        "--num_preds_to_save",
        type=int,
        default=0,
        help="At the end of training, save N preds for each query. "
        "Try with a small number like 3",
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="When saving preds (if num_preds_to_save != 0) save only "
        "preds for difficult queries, i.e. with uncorrect first prediction",
    )
    # Paths parameters
    parser.add_argument(
        "--log_dir",
        type=str,
        default="default",
        help="name of directory on which to save the logs, under logs/log_dir",
    )
    parser.add_argument(
        "--real_train_dir",
        type=str,
        required=True,
        help="path of directory with real images",
    )
    parser.add_argument(
        "--synt_train_dir",
        type=str,
        required=True,
        help="path of directory with synthetic images",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="path of directory with test datasets",
    )

    args = parser.parse_args()

    args.log_dir = Path("logs") / args.log_dir

    return args
