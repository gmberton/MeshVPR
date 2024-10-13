import sys
import torch
from loguru import logger
from datetime import datetime

import test
import parser
import vpr_models
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

THRESHOLDS = [10, 25, 50, 100, 200, 500, 1000]

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
args.log_dir = args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
logger.add(args.log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
logger.add(args.log_dir / "debug.log", level="DEBUG")
logger.info(" ".join(sys.argv))
logger.info(f"Arguments: {args}")
logger.info(f"The outputs are being saved in {args.log_dir}")

#### MODELS
real_model = vpr_models.get_model(args.method).to(args.device)
synt_model = vpr_models.get_model(args.method).to(args.device)

if args.resume_model is not None:
    logger.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    synt_model.load_state_dict(model_state_dict)
else:
    logger.info(
        "WARNING: You didn't provide a path to resume the model (--resume_model parameter). "
        + "Evaluation will be computed using randomly initialized weights."
    )

for dataset_name in ["synt_berlin", "synt_paris", "synt_melbourne"]:
    test_dataset = TestDataset(
        dataset_dir=args.test_dir,
        dataset_name=dataset_name,
    )
    all_recalls, all_recalls_str = test.test(
        args,
        test_dataset,
        real_model,
        synt_model,
        args.num_preds_to_save,
        preds_folder_name=test_dataset.dataset_name,
        positive_dist_threshold=THRESHOLDS,
    )
    for threshold in THRESHOLDS:
        logger.info(
            f"{test_dataset} thresh={threshold: >4}: {all_recalls_str[threshold]}"
        )

logger.info(
    f"Experiment finished (without any errors), in total in {str(datetime.now() - start_time)[:-7]}"
)
