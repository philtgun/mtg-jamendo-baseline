import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from torchmetrics import AUROC, AveragePrecision, MetricCollection

import wandb

from .dataset import MtgJamendoDataset
from .layers import ConvUnit
from .losses import FocalLoss
from .results import add_to_output


class FullyConvNet(pl.LightningModule):
    INPUT_SIZE = [96, 1366]

    def __init__(self, num_classes, lr, optimizer, loss):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer

        if loss == 'bce':
            self.criterion = nn.BCELoss()
        elif loss == 'focal':
            self.criterion = FocalLoss(from_logits=False)
        else:
            raise ValueError(f'Invalid loss: {loss}')

        self.frontend = nn.Sequential(
            nn.BatchNorm2d(1),
            ConvUnit(1, 64, pool_size=(2, 4)),
            ConvUnit(64, 128, pool_size=(2, 4)),
            ConvUnit(128, 128, pool_size=(2, 4)),
            ConvUnit(128, 128, pool_size=(3, 5)),
            ConvUnit(128, 64, pool_size=(4, 4)),
        )
        self.backend = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        metrics = MetricCollection({
            'roc_auc': AUROC(num_classes=num_classes, average='macro'),
            'pr_auc': AveragePrecision(num_classes=num_classes, average='macro'),
        })
        self.metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.frontend(x)
        x = x.view(-1, 64)
        x = self.backend(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        self.metrics.update(y_hat, y.int())
        return y_hat, y

    def validation_epoch_end(self, validation_step_outputs):
        for metric, value in self.metrics.compute().items():
            self.log(metric, value)
        self.metrics.reset()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        self.test_metrics.update(y_hat, y.int())
        return y_hat, y

    def test_epoch_end(self, test_step_outputs):
        for metric, value in self.test_metrics.compute().items():
            self.log(metric, value)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        raise ValueError(f'Invalid optimizer: {self.optimizer}')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer')
        parser.add_argument('--loss', choices=['bce', 'focal'], default='bce', help='Loss function')
        return parser


def train_main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train the model on MTG-Jamendo Dataset')

    parser.add_argument('data_path', help='Path to the data directory')
    parser.add_argument('repo_path', help='Path to the directory with mtg-jamendo repository')
    parser.add_argument('--subset', choices=['autotagging', 'autotagging_genre', 'autotagging_instrument',
                                             'autotagging_moodtheme', 'autotagging_top50tags'], default='autotagging',
                        help='Dataset subset')
    parser.add_argument('--split', default=0, help='Split index')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=32, help='Size of input batches')
    parser.add_argument('--sampling-strategy', choices=MtgJamendoDataset.SAMPLING_STRATEGY, default='center',
                        help='The sampling strategy for the input chunks')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of worker processes for data loader')
    parser.add_argument('--output-path', help='CSV file to add the run results to (split, subset, ROC_AUC, PR_AUC)')
    parser.add_argument('--models-dir', help='Directory to save models in')

    parser = FullyConvNet.add_model_specific_args(parser)
    parser = pl.trainer.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    wandb.init(config=args)
    pl.seed_everything(args.seed)

    train_list = MtgJamendoDataset.get_tsv_file(args.repo_path, args.subset, 'train', args.split)
    validation_list = MtgJamendoDataset.get_tsv_file(args.repo_path, args.subset, 'validation', args.split)
    test_list = MtgJamendoDataset.get_tsv_file(args.repo_path, args.subset, 'test', args.split)
    # tags_file = MtgJamendoDataset.get_tags_file(args.repo_path, 'top50')

    input_size = FullyConvNet.INPUT_SIZE[1]
    train_dataset = MtgJamendoDataset(args.data_path, train_list, input_size,  # tags_file,
                                      sampling_strategy=args.sampling_strategy)
    validation_dataset = MtgJamendoDataset(args.data_path, validation_list, input_size,  # tags_file,
                                           sampling_strategy=args.sampling_strategy)
    test_dataset = MtgJamendoDataset(args.data_path, test_list, input_size,  # tags_file,
                                     sampling_strategy=args.sampling_strategy)

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(monitor='val_roc_auc', mode='max', dirpath=args.models_dir,
                                          filename=f'{args.subset}-split{args.split}.ckpt')

    num_classes = train_dataset.y.shape[1]
    model = FullyConvNet(num_classes, args.lr, args.optimizer, args.loss)
    logger.watch(model)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, num_sanity_val_steps=-1,
                                            callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, validation_dataloader)
    results = trainer.test(dataloaders=test_dataloader)
    if args.output_path is not None:
        for result in results:
            add_to_output(args.output_path, [args.split, args.subset, result['test_roc_auc'], result['test_pr_auc']])


if __name__ == '__main__':
    train_main()
