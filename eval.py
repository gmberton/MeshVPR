import sys
import torch
import logging
from datetime import datetime

import test
import parser
import commons
import vpr_models
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

THRESHOLDS = [10, 25, 50, 100, 200, 500, 1000]

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
args.log_dir = args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
commons.setup_logging(args.log_dir, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.log_dir}")

#### MODELS
real_model = vpr_models.get_model(args.method).to(args.device)
synt_model = vpr_models.get_model(args.method).to(args.device)

if args.resume_model is not None:
    logging.info(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    synt_model.load_state_dict(model_state_dict)
else:
    logging.info(
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
        logging.info(
            f"{test_dataset} thresh={threshold: >4}: {all_recalls_str[threshold]}"
        )

logging.info(
    f"Experiment finished (without any errors), in total in {str(datetime.now() - start_time)[:-7]}"
)
