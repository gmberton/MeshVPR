import faiss
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
from loguru import logger
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset

import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20, 25, 50, 100]


def test(
    args: Namespace,
    eval_ds: Dataset,
    real_model,
    synt_model,
    num_preds_to_save: int = 0,
    preds_folder_name="default",
    positive_dist_threshold=[10, 25, 50, 100, 200, 500, 1000],
) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""

    real_model = real_model.eval()
    synt_model = synt_model.eval()
    faiss_index = faiss.IndexFlatL2(real_model.desc_dim)

    with torch.no_grad():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.num_db)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
        )
        for db_images, indices, paths in tqdm(database_dataloader, ncols=120):
            descriptors = synt_model(db_images.to(args.device))
            faiss_index.add(descriptors.cpu().numpy())

        logger.debug(
            "Extracting queries descriptors for evaluation/testing using batch size 1"
        )
        queries_subset_ds = Subset(
            eval_ds, list(range(eval_ds.num_db, eval_ds.num_db + eval_ds.num_q))
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1
        )
        queries_descriptors = []
        for q_images, indices, paths in tqdm(queries_dataloader, ncols=120):
            descriptors = real_model(q_images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            queries_descriptors.append(descriptors)
        queries_descriptors = np.concatenate(queries_descriptors)

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    all_recalls = {}
    all_recalls_str = {}
    # Thresholds are in meters
    for positive_dist_threshold in [10, 25, 50, 100, 200, 500, 1000]:
        positives_per_query = eval_ds.get_positives(positive_dist_threshold)
        recalls = np.zeros(len(RECALL_VALUES))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(RECALL_VALUES):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_q and multiply by 100, so the recalls are in percentages
        recalls = recalls / eval_ds.num_q * 100
        recalls_str = ", ".join(
            [f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)]
        )
        all_recalls[positive_dist_threshold] = recalls
        all_recalls_str[positive_dist_threshold] = recalls_str

        # Save visualizations of predictions
        if num_preds_to_save != 0 and positive_dist_threshold == 100:
            # For each query save num_preds_to_save predictions
            visualizations.save_preds(
                predictions[:, :num_preds_to_save],
                eval_ds,
                args.log_dir,
                args.save_only_wrong_preds,
                preds_folder_name=preds_folder_name,
                positive_dist_threshold=positive_dist_threshold,
            )

    return all_recalls, all_recalls_str
