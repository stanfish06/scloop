# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE

from .data_modules import nnRegressorDataModule


class NeuralODEregressor(pl.LightningModule):
    data: nnRegressorDataModule
    t_span: torch.Tensor
    input_dim: int
    output_dim: int
    do_validation: bool
    n_hidden: int
    n_layers: int
    activation_fn: nn.Module
    solver: str
    solver_adjoint: str
    atol_adjoint: tuple[float]
    rtol_adjoint: tuple[float]
    lr: float
    weight_decay: float
    trainer: pl.Trainer | None
    check_val_every_n_epoch: int

    def __init__(
        self,
        data: nnRegressorDataModule,
        t_span: torch.Tensor,
        n_hidden=64,
        n_layers=1,
        solver="rk4",
        solver_adjoint="dopri5",
        atol_adjoint=1e-4,
        rtol_adjoint=1e-4,
        lr=1e-3,
        weight_decay=1e-4,
        check_val_every_n_epoch=5,
    ):
        super().__init__()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.data = data
        self.t_span = t_span.to(device)
        self.input_dim = data.input_dim
        self.output_dim = data.output_dim
        self.do_validation = data.do_validation
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation_fn = nn.Tanh()
        self.solver = solver
        self.solver_adjoint = solver_adjoint
        self.atol_adjoint = (atol_adjoint,)
        self.rtol_adjoint = (rtol_adjoint,)
        self.lr = lr
        self.weight_decay = weight_decay
        self.trainer = None
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.save_hyperparameters("n_hidden", "n_layers")

        layers_dims = (
            [self.input_dim] + self.n_layers * [self.n_hidden] + [self.output_dim]
        )

        layers = []
        for i, (n_in, n_out) in enumerate(
            zip(layers_dims[:-1], layers_dims[1:], strict=False)
        ):
            layers.append(
                (
                    f"layer_{i}",
                    nn.Sequential(
                        nn.Linear(n_in, n_out, bias=True),
                        self.activation_fn
                        if i < len(layers_dims) - 2
                        else nn.Identity(),  # No activation on final layer
                    ),
                )
            )

        self.model = NeuralODE(
            nn.Sequential(OrderedDict(layers)),
            sensitivity="adjoint",
            solver=self.solver,
            solver_adjoint=self.solver_adjoint,
        )

    def forward(self, x, t_span):
        return self.model(x, t_span)

    def compute_loss(self, y_hat, y):
        mse_loss = F.mse_loss(y_hat, y)
        return mse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        # select last point of solution trajectory
        y_hat = y_hat[-1]
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = self.compute_loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.forward(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = self.compute_loss(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):  # type: ignore[attr-defined]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" if self.do_validation else "train_loss",
                "frequency": self.check_val_every_n_epoch,
            },
            "monitor": "val_loss" if self.do_validation else "train_loss",
        }

    def predict_step(self, batch, batch_idx):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return self.forward(x, self.t_span)

    def fit(self, max_epochs: int = 100):
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            logger=False,
            enable_checkpointing=True,
        )
        self.trainer.fit(self, self.data)

    def predict(self):
        if self.trainer is None:
            raise RuntimeError("Model has not been trained")

        batch_predictions = self.trainer.predict(self, self.data)
        t_eval = batch_predictions[0][0]  # type: ignore[attr-defined]
        all_trajectories = torch.cat([pred[1] for pred in batch_predictions], dim=1)  # type: ignore[attr-defined]
        final_snapshots = all_trajectories[-1, :, :]

        t_eval_np = t_eval.detach().cpu().numpy()
        trajectories_np = all_trajectories.detach().cpu().numpy()
        final_np = final_snapshots.detach().cpu().numpy()

        return (t_eval_np, trajectories_np, final_np)

    def predict_new(self, x: np.ndarray):
        import numpy as np

        if self.trainer is None:
            raise RuntimeError("Model has not been trained")

        x_tensor = torch.from_numpy(x.astype(np.float32))
        self.eval()
        with torch.no_grad():
            _, trajectory = self.forward(x_tensor, self.t_span)
            y_pred = trajectory[-1]
        return y_pred.detach().cpu().numpy()
