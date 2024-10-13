import os
import sys
import torch
import torchmetrics
from glob import glob
from tqdm import tqdm
from loguru import logger
from datetime import datetime

import test
import util
import parser
import vpr_models
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.log_dir = args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
logger.add(args.log_dir / "debug.log", level="DEBUG")
logger.add(args.log_dir / "info.log", level="INFO")
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

#### DATA
train_ds = TrainDataset(
    real_dir=args.real_train_dir,
    synt_dir=args.synt_train_dir,
    train_on_southern_half=args.train_on_southern_half,
)

dataloader = torch.utils.data.DataLoader(
    train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
)

val_ds = TestDataset(
    dataset_dir=args.test_dir,
    dataset_name="val_set",
)

test_sets_names = [os.path.basename(n) for n in sorted(glob(args.test_dir + "/*"))]
logger.info(f"Found {len(test_sets_names)} test sets, namely {test_sets_names}")

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=synt_model.parameters(), lr=args.lr)

best_val_recall1 = 0
num_epoch = 0

for num_epoch in range(args.num_epochs):
    epoch_start_time = datetime.now()
    real_model = real_model.eval()
    synt_model = synt_model.train()
    epoch_losses = torchmetrics.MeanMetric()
    tqdm_bar = tqdm(
        dataloader, ncols=120, total=min(len(dataloader), args.iterations_per_epoch)
    )
    for n_iter, (
        real_images,
        synt_images,
        real_paths,
        synt_paths,
        indexes,
    ) in enumerate(tqdm_bar):
        if n_iter >= args.iterations_per_epoch:
            break

        with torch.autocast(args.device):
            with torch.inference_mode():
                real_descs = real_model(real_images.to(args.device))
            synt_descs = synt_model(synt_images.to(args.device))
            loss = criterion(real_descs.clone(), synt_descs)

        loss.backward()
        optim.step()
        optim.zero_grad()

        epoch_losses.update(loss.item())
        cur_mean_loss = epoch_losses.compute()
        tqdm_bar.desc = f"cur_mean_loss: {cur_mean_loss:.10f}"

    recalls, recalls_str = test.test(
        args, val_ds, real_model, synt_model, positive_dist_threshold=[100]
    )
    logger.info(
        f"Epoch {num_epoch:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"loss = {cur_mean_loss:.10f}, {val_ds}: {recalls_str[100][:20]}"
    )

    is_best = recalls[100][0] > best_val_recall1
    best_val_recall1 = max(recalls[100][0], best_val_recall1)

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(
        {
            "num_epoch": num_epoch + 1,
            "synt_model_state_dict": synt_model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "best_val_recall1": best_val_recall1,
        },
        is_best,
        args.log_dir,
    )

logger.info(
    f"Trained for {num_epoch+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}"
)

for dataset_name in test_sets_names:
    test_dataset = TestDataset(
        dataset_dir=args.test_dir,
        dataset_name=dataset_name,
    )
    recalls, recalls_str = test.test(
        args,
        test_dataset,
        real_model,
        synt_model,
        args.num_preds_to_save,
        preds_folder_name=test_dataset.dataset_name,
        positive_dist_threshold=[100],
    )
    logger.info(f"{test_dataset}: {recalls_str[100]}")

logger.info(
    f"Experiment finished (without any errors), in total in {str(datetime.now() - start_time)[:-7]}"
)
