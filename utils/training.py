import numpy as np
import torch.cuda
from pytorch_lightning import Trainer
import os
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.data import GymDataModule, ActiveSensingDataset

DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PerceptionModelTrainer(Trainer):
    """
    # TODO: add description
    """

    def __init__(self, beta_schedule=None,
                 rec_loss_scale=1.0,
                 *args, **kwargs):
        super(PerceptionModelTrainer, self).__init__(*args, **kwargs)

        self.rec_loss_scale = rec_loss_scale

        if beta_schedule is None:
            beta_schedule = np.ones(self.max_epochs)

        self.beta_schedule = beta_schedule


def train_perception_model(model, data,
                           n_epochs=50,
                           batch_size=32,
                           beta_schedule=None,
                           rec_loss_scale=1.0,
                           grad_clip_value=0,
                           checkpoint_every=10,
                           val_every=5,
                           train_size=8000,
                           val_frac=0.1,
                           log_dir=None,
                           exp_name='',
                           dataset_class=ActiveSensingDataset,
                           monitored_loss='total_loss'):
    if log_dir is None:
        log_dir = PROJECT_DIR
        log_dir = os.path.join(log_dir, 'perception-training-results')

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if not os.path.isdir(os.path.join(log_dir, exp_name)):
        os.makedirs(os.path.join(log_dir, exp_name))

    # ###### CREATE THE LOGGERS ####### #
    csv_logger = loggers.CSVLogger(save_dir=log_dir, name=exp_name, version='')
    tb_logger = loggers.TensorBoardLogger(save_dir=log_dir, name=exp_name, version='')

    # ######## CHECKPOINTING ############ #
    chkpt_callback = ModelCheckpoint(
        every_n_epochs=checkpoint_every,
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=1,
        monitor=monitored_loss,
        dirpath=os.path.join(log_dir, exp_name),
        filename='{epoch:02d}-{vae_loss:.4f}'
    )

    # ####### TRAINER ########## #
    trainer = PerceptionModelTrainer(
        # perception model specific arguments
        beta_schedule=beta_schedule,
        rec_loss_scale=rec_loss_scale,

        # training logistics
        accelerator=DEVICE,
        check_val_every_n_epoch=val_every,
        default_root_dir=log_dir,
        logger=[csv_logger, tb_logger],
        callbacks=[chkpt_callback],

        # training params
        max_epochs=n_epochs,
        gradient_clip_val=grad_clip_value,

        # for debugging
        detect_anomaly=True
    )

    # ####### INITIALIZE THE DATA MODULE ####### #
    if type(data) is str:
        data_dir = data
        data_dict = None
    elif type(data) is dict:
        data_dict = data
        data_dir = None
    else:
        raise Exception("data must either be a dict or a string indicating the path to the data file")
    data_module = GymDataModule(data_dir=data_dir, data_dict=data_dict, batch_size=batch_size,
                                train_size=train_size, val_frac=val_frac, dataset_class=dataset_class)

    # ######## FIT ############## #
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("Done!")
