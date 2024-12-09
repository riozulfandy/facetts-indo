import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import os
import copy
import torch
from config import ex
from model.face_tts import FaceTTS
from data import _datamodules

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    pl.seed_everything(_config["seed"])
    
    dm = _datamodules["dataset_" + _config["dataset"]](_config)
    
    
    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
    )
    
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    model = FaceTTS(_config)
    
    # Manually load state dict if checkpoint only contains state_dict
    ckpt_path = _config.get("resume_from")
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)
    
    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]
    
    num_gpus = (
        _config["num_gpus"] 
        if isinstance(_config["num_gpus"], int) 
        else len(_config["num_gpus"])
    )
    
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        strategy=DDPPlugin(gradient_as_bucket_view=True, find_unused_parameters=True),
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        weights_summary="top",
        val_check_interval=_config["val_check_interval"],
    )
    
    if not _config.get("test_only", False):
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
