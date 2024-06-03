import os
import torch
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
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir=None, state_dict=None):
        if getattr(self.args, 'tune_mlp_adapter', False) and getattr(self.args, 'freeze_backbone', False):
            pass
        else:
            super()._save(output_dir, state_dict)


class RankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        inputs_w_content = inputs.pop("inputs_w_content", None)
        inputs_wo_content = inputs.pop("inputs_wo_content", None)
        extra_text_inputs = inputs.pop("extra_text_inputs", dict())

        if inputs_wo_content is not None:
            outputs1 = model(
                **inputs_wo_content,
                **extra_text_inputs,
                **inputs,
            )
        if inputs_w_content is not None:
            outputs2 = model(
                **inputs_w_content,
                **extra_text_inputs,
                **inputs,
            )

        if inputs_wo_content is not None:
            loss1, logits1 = outputs1.loss, outputs1.logits
        else:
            loss1, logits1 = None, None
        if inputs_w_content is not None:
            loss2, logits2 = outputs2.loss, outputs2.logits
        else:
            loss2, logits2 = None, None

        if self.args.kl_loss_weight > 0 and loss1 is not None and loss2 is not None:
            loss = self.args.loss1_weight * loss1 + self.args.loss2_weight * loss2
            kl_loss = torch.nn.functional.kl_div(
                input=torch.log_softmax(logits1, dim=-1),
                target=torch.log_softmax(logits2, dim=-1),
                log_target=True,
                reduction="batchmean",
            )
            loss += self.args.kl_loss_weight * kl_loss
        else:
            loss = self.args.loss1_weight * loss1 + self.args.loss2_weight * loss2 \
                if (loss1 and loss2) else (loss1 or loss2)

        outputs = {
            "loss": loss,
            "logits1": logits1,
            "logits2": logits2,
        }

        return (loss, outputs) if return_outputs else loss
