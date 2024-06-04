
import torch
from pathlib import Path


def save_checkpoint(state: dict, is_best: bool, log_dir: Path,
                    ckpt_filename: str = "last_checkpoint.pth"):
    checkpoint_path = log_dir / ckpt_filename
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state["synt_model_state_dict"], f"{log_dir}/best_model.pth")
