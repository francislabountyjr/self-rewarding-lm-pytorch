from copy import deepcopy
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Module, Dropout
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from accelerate import Accelerator

from beartype import beartype
from beartype.typing import Optional, Callable

from einx import get_at

from pytorch_custom_utils import get_adam_optimizer, OptimizerWithWarmupSchedule

from pytorch_custom_utils.accelerate_utils import (
    auto_unwrap_model,
    model_forward_contexts,
)

from torchtyping import TensorType

# helper functions


def exists(v):
    return v is not None


def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False


def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob


def adam_optimizer_with_linear_decay(
    model: Module,
    start_learning_rate: float,
    end_learning_rate: float,
    num_decay_steps: int,
    accelerator: Accelerator,
    weight_decay: float,
    adam_kwargs: dict = dict(),
) -> OptimizerWithWarmupSchedule:
    adam = get_adam_optimizer(
        model.parameters(), lr=start_learning_rate, wd=weight_decay
    )

    return OptimizerWithWarmupSchedule(
        optimizer=adam,
        accelerator=accelerator,
        scheduler=LinearLR,
        scheduler_kwargs=dict(
            start_factor=1.0,
            end_factor=end_learning_rate / start_learning_rate,
            total_iters=num_decay_steps,
        ),
    )


# early stopping


@dataclass
class EarlyStopperReturn:
    should_stop: bool
    score: float


class EarlyStopper(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        calculate_should_stop: Callable[..., bool] = lambda past_scores, score: len(
            past_scores
        )
        > 0
        and score < past_scores[-1],
        early_stop_checkpoint_folder: str = "./early-stop-checkpoint",
    ):
        super().__init__()
        self.model = model
        self.scores = []
        self.calculate_should_stop = calculate_should_stop

        self.val_dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.early_stop_checkpoint_folder = Path(early_stop_checkpoint_folder)
        self.early_stop_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    def save(self, path: str, overwrite: bool = False):
        if not self.accelerator.is_main_process:
            return

        path = self.checkpoints_folder / path

        assert not path.exists() or overwrite, f"file already exists"

        pkg = dict(model=self.model.state_dict())

        torch.save(pkg, str(path))

    @torch.no_grad()
    def forward(self) -> EarlyStopperReturn:
        self.model.eval()

        raise NotImplementedError

        should_stop = self.calculate_should_stop(self.scores, score)
        self.scores.append(score)

        if should_stop:
            prev_checkpoint_filename = f"model.ckpt.{len(self.scores) - 1}.pt"
            ckpt_path = self.early_stop_checkpoint_folder / prev_checkpoint_filename

            pkg = torch.load(str(ckpt_path))

            self.model.load_state_dict(pkg["model"])
        else:
            checkpoint_filename = f"model.ckpt.{len(self.scores)}.pt"
            ckpt_path = self.early_stop_checkpoint_folder / checkpoint_filename
            self.save(str(ckpt_path))

        return EarlyStopperReturn(score, should_stop)


# dataset from two memmap numpy file

# preferred and unpreferred sequences of shape - (<num samples>, <preference (2) - preferred followed by unpreferred>, <seq length>)
# prompt length (<num samples>,)


class DPODataset(Dataset):
    def __init__(self, preference_seq_memmap_file: str, prompt_len_memmap_file: str):
        assert Path(preference_seq_memmap_file).exists()
        assert Path(prompt_len_memmap_file).exists()

        self.paired_sequences = open_memmap(
            preference_seq_memmap_file, dtype="int", mode="r"
        )
        self.prompt_len = open_memmap(prompt_len_memmap_file, dtype="int", mode="r")

        self.seq_len = self.paired_sequences.shape[1]
        assert self.paired_sequences.shape[0] == self.prompt_len == [0]

    def __len__(self):
        return self.paired_sequences.shape[0]

    def __getitem__(self, idx):
        sequences = self.paired_sequences[idx]
        prompt_lens = self.prompt_len[idx]

        preferred_seq, unpreferred_seq = self.paired_sequences.unbind(dim=1)

        return preferred_seq, unpreferred_seq, prompt_lens


# main class


class CPO(Module):
    def __init__(
        self,
        model: Module,
        *,
        beta=0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.beta = beta
        self.pad_id = pad_id

    def parameters(self):
        return self.model.parameters()

    def update_reference_model_with_policy(self):
        pass

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.pad_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.pad_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    @property
    def device(self):
        return next(self.parameters()).device

    @autocast(enabled=False)
    def forward(
        self,
        preferred_seq: TensorType["b", "n", int],
        unpreferred_seq: TensorType["b", "n", int],
        prompt_mask: Optional[TensorType["b", "n", bool]] = None,
        preferred_seq_mask: Optional[TensorType["b", "n", bool]] = None,
    ):
        """
        b - batch
        n - sequence length
        """
        max_length = max(preferred_seq.shape[1], unpreferred_seq.shape[1])
        len_preferred = preferred_seq.shape[1]

        preferred_seq_padded = F.pad(
            preferred_seq, (0, max_length - preferred_seq.shape[1])
        )
        unpreferred_seq_padded = F.pad(
            unpreferred_seq, (0, max_length - unpreferred_seq.shape[1])
        )

        concatenated_seq = torch.cat(
            [preferred_seq_padded, unpreferred_seq_padded], dim=0
        )

        if prompt_mask is not None:
            prompt_mask_padded = F.pad(
                prompt_mask, (0, max_length - prompt_mask.shape[1]), value=False
            )
            concatenated_mask = torch.cat(
                [prompt_mask_padded, prompt_mask_padded], dim=0
            )
        else:
            concatenated_mask = None

        outputs = self.model(concatenated_seq, attention_mask=concatenated_mask)

        def cross_entropy_loss(logits, labels):
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss

        all_logits = outputs.logits.to(torch.float32)
        if isinstance(outputs, dict) and "loss" not in outputs:
            labels = concatenated_seq.to(torch.long)
            loss_chosen = cross_entropy_loss(
                all_logits[:len_preferred], labels[:len_preferred]
            )
            # loss_reject = cross_entropy_loss(all_logits[len_chosen:], labels[len_chosen:])
            # clm_loss = 0.5 * (loss_chosen + loss_reject)
            clm_loss = loss_chosen
        else:
            clm_loss = outputs.loss

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_seq,
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_preferred]
        rejected_logps = all_logps[len_preferred:]

        logits = chosen_logps - rejected_logps

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        loss = losses.mean() + clm_loss

        metrics = {}
        chosen_logits = all_logits[:len_preferred]
        rejected_logits = all_logits[len_preferred:]
        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics["rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics["rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics["rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics["logps/rejected"] = rejected_logps.detach().cpu().mean()
        metrics["logps/chosen"] = chosen_logps.detach().cpu().mean()
        metrics["logits/rejected"] = rejected_logits.detach().cpu().mean()
        metrics["logits/chosen"] = chosen_logits.detach().cpu().mean()
        metrics["clm_loss"] = clm_loss.cpu().mean()
        metrics["cpo_loss"] = losses.cpu().mean()
        metrics["loss"] = loss.cpu().mean()

        return loss, metrics


# trainer class


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class CPOTrainer(Module):
    @beartype
    def __init__(
        self,
        cpo: CPO,
        *,
        accelerator: Accelerator,
        batch_size: int = 16,
        num_decay_steps: int = 1000,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        val_dataset: Optional[Dataset] = None,
        start_learning_rate: float = 1e-6,
        end_learning_rate: float = 1e-7,
        adam_kwargs: dict = dict(),
        early_stopper: Optional[EarlyStopper] = None,
        dropout: float = 0.1,
        check_early_stop_every: int = 200,
        log_every: int = 10,
    ):
        super().__init__()
        set_dropout_(cpo, dropout)

        self.accelerator = accelerator
        self.model = accelerator.prepare(cpo)

        self.batch_size = batch_size

        self.optimizer = adam_optimizer_with_linear_decay(
            cpo,
            start_learning_rate,
            end_learning_rate,
            num_decay_steps=num_decay_steps,
            accelerator=accelerator,
            weight_decay=weight_decay,
            adam_kwargs=adam_kwargs,
        )

        self.early_stopper = None
        if self.is_main:
            self.early_stopper = early_stopper

        self.check_early_stop_every = check_early_stop_every

        self.val_dataloader = None
        if exists(val_dataset):
            self.val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, drop_last=True, shuffle=True
            )

        self.log_every = log_every

        self.steps = 0
        self.register_buffer("break_signal", torch.tensor(0.0))

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def forward(self, train_self_reward_dataset: Dataset):
        train_dataloader = DataLoader(
            train_self_reward_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )
        iter_dl = cycle(train_dataloader)

        while True:
            self.model.train()

            batch = next(iter_dl)

            cpo_loss, metrics = self.model(batch)
            self.accelerator.backward(cpo_loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1

            self.accelerator.wait_for_everyone()

            self.accelerator.log(metrics, step=self.steps)

            if not (self.steps % self.check_early_stop_every):
                if self.is_main and self.early_stopper():
                    self.break_signal.copy_(1.0)
                    dist.all_reduce(self.break_signal)

                if self.break_signal.item() == 1:
                    break

            self.accelerator.wait_for_everyone()

        raise NotImplementedError
