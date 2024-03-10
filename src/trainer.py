import os
import torch
from typing import Dict, List, Optional, Tuple
from torch.nn.modules import Module

from transformers import Trainer as HFTrainer

from utils import get_adapter_state_maybe_zero_3


class Trainer(HFTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mlp_adapter', False) and getattr(self.args, 'freeze_backbone', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['projector']

            weight_to_save = get_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'projector.bin'))
        else:
            super()._save_checkpoint(model.get_model(), trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mlp_adapter', False) and getattr(self.args, 'freeze_backbone', False):
            pass
        else:
            super()._save(output_dir, state_dict)