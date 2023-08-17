"""Custom trainer"""

import os
from typing import Any, Iterable, Optional, Union

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from diffuser.diffuser import Diffuser
from utils.utils import Params


class CustomTrainer:
    """Custom trainer using acclerate for ddp, fsdp and mixed precision training
    Args:
        params: Hyper-parameters
    """

    def __init__(
        self,
        params: Params,
    ) -> None:
        self.params = params

        self.accelerate = Accelerator(
            project_dir=params.out_dir,
            log_with="tensorboard",
            gradient_accumulation_steps=params.grad_accum_steps,
        )
        self.overall_step = 0
        self.current_epoch = 0

        self.should_stop = False

    def fit(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: The torch Module to train.
            optimizer: Torch optimzer
            scheduler: Torch scheduler
            train_loader: The training dataloader. Has to be an iterable returning batches.
            val_loader: The validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
        """
        if self.params.checkpoint_steps != "epoch":
            try:
                self.params.checkpoint_steps = int(self.params.checkpoint_steps)
            except ValueError as err:
                raise err

        self.accelerate.init_trackers(project_name=self.params.tb_dir)

        if val_loader is not None:
            model, optimizer, train_loader, val_loader, scheduler = self.accelerate.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )

        else:
            model, optimizer, train_loader, scheduler = self.accelerate.prepare(
                model, optimizer, train_loader, scheduler
            )

        diffuser = Diffuser(self.params.timesteps, self.accelerate.device)

        if self.params.resume:
            # Get the most recent checkpoint
            dirs = [file_.name for file_ in os.scandir(self.params.out_dir) if file_.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                self.current_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = 0
            else:
                resume_step = int(training_difference.replace("step_", ""))
                self.overall_step = resume_step
                self.current_epoch = resume_step // len(train_loader)
                resume_step -= self.current_epoch * len(train_loader)

            self.should_stop = (self.current_epoch >= self.params.num_epochs) or (
                hasattr(self.params, "max_steps") and self.overall_step >= self.params.max_steps
            )

            train_loader = self.accelerate.skip_first_batches(train_loader, resume_step)

        while not self.should_stop:
            self.train_loop(model, optimizer, train_loader, scheduler, diffuser)

            if self.current_epoch % self.params.validation_frequency == 0:
                self.val_loop(model, val_loader)

            if self.params.checkpoint_steps == "epoch":
                if (
                    (self.current_epoch + 1) % self.params.save_epoch_freq == 0
                    or self.current_epoch == (self.params.num_epochs - 1)
                ):
                    output_dir = os.path.join(self.params.out_dir, f"epoch_{self.current_epoch}")
                    self.accelerate.save_state(output_dir)
            self.current_epoch += 1

            # stopping condition on epoch level
            self.should_stop = self.current_epoch >= self.params.num_epochs

        self.accelerate.end_training()
        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        diffuser: Diffuser,
    ) -> None:
        """Training loop"""
        model.train()
        iterable = self.progbar_wrapper(
            train_loader, total=len(train_loader), desc=f"Epoch {self.current_epoch}"
        )
        total_loss = 0.0
        for imgs, labels in iterable:
            with self.accelerate.accumulate(model):
                loss = self.train_step(model, imgs, labels, diffuser)
                self.accelerate.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.overall_step += 1

                total_loss += loss.detach().item()

            self._format_iterable(iterable, loss.detach(), "Train")

            if isinstance(self.params.checkpoint_steps, int):
                if self.overall_step % self.params.checkpoint_steps == 0:
                    output_dir = os.path.join(self.params.out_dir, f"step_{self.overall_step}")
                    self.accelerate.save_state(output_dir)

            # stopping criterion on step level
            if hasattr(self.params, "max_steps") and self.overall_step >= self.params.max_steps:
                self.should_stop = True
                break

        avg_loss = total_loss / len(train_loader)
        self.accelerate.print(f"Average train loss: {avg_loss:.3f}")
        self.accelerate.log(
            {"Train_loss": avg_loss, "Learning_rate": scheduler.get_last_lr()[0]},
            step=self.current_epoch,
        )

    def train_step(
        self, model: torch.nn.Module, imgs: torch.Tensor, labels: torch.Tensor, diffuser: Diffuser
    ) -> torch.Tensor:
        """Training step"""
        batch_size = imgs.shape[0]

        # randomly mask out context
        context_mask = torch.bernoulli(torch.zeros(batch_size) + 0.9).to(self.accelerate.device)
        labels = labels * context_mask.unsqueeze(-1)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        noise = torch.randn_like(imgs)
        time = torch.randint(
            1, self.params.timesteps + 1, (batch_size,), device=self.accelerate.device
        )
        x_pert = diffuser.q_sample(imgs, time, noise)

        # use network to recover noise
        pred_noise = model(x_pert, time / self.params.timesteps, labels)

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return loss

    def val_loop(
        self,
        model: torch.nn.Module,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        """Validation loop"""
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        model.eval()
        iterable = self.progbar_wrapper(val_loader, total=len(val_loader))
        total_loss = 0.0
        with torch.no_grad():
            for imgs, labels in iterable:
                preds = model(imgs, labels)

                loss = torch.nn.functional.mse_loss(preds, labels)
                total_loss += loss.item()

                self._format_iterable(iterable, loss.detach(), "Val")

            avg_loss = total_loss / len(val_loader)
            self.accelerate.print(f"Average Validation loss: {avg_loss:.3f}")
            self.accelerate.log({"Val_loss": avg_loss}, step=self.current_epoch)

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any) -> Iterable:
        """Wraps the iterable with tqdm for global rank zero.
        Args:
            iterable: The iterable to wrap with tqdm
            total: The total length of the iterable, necessary in case the number of batches was
                limited.
        Returns:
            Iterable or progressbar wrapped iterable
        """
        if self.accelerate.process_index == 0:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    @staticmethod
    def _format_iterable(
        prog_bar: Iterable,
        candidates: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]],
        prefix: str,
    ) -> None:
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: Progressbar (on global rank zero) or an iterable (every other rank).
            candidates: Values to add as postfix strings to the progressbar.
            prefix: Prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {candidates.item():.3f}"
            elif isinstance(candidates, dict):
                for k, val in candidates.items():
                    postfix_str += f" {prefix}_{k}: {val.item():.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
