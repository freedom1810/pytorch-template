import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, SaveCsv
from tqdm import tqdm
from torch.cuda import amp
# from trainer.mix import mixup
import pandas as pd
from .ema import  ModelEMA
import torch.nn as nn
import os
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
        self, 
        model, 
        train_criterion, 
        val_criterion, 
        metric_ftns, 
        optimizer, 
        config, 
        device,
        data_loader,
        valid_data_loader=None, 
        lr_scheduler = None,
        len_epoch=None, 
        fold_idx=0, 
        warmup=0
        ):

        super().__init__(model, train_criterion, val_criterion, metric_ftns, optimizer, config, fold_idx, warmup)
        self.config = config
        self.device = device

        self.fp16 = self.config['fp16']
        self.scaler = amp.GradScaler(enabled=self.fp16)

        if self.config['ema']:
            self.ema =  ModelEMA(self.model)
        else:
            self.ema = None
    
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        
        self.lr_scheduler = lr_scheduler

        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = int(len(data_loader) // 5)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.csv_save = SaveCsv(self.checkpoint_dir, fold_idx)

    def _save_only_top_10(self):
        top_n_model = 10
        self.df = pd.DataFrame(self.csv_save.df)
        top_auc = self.df.sort_values(by='val_roc_auc', ascending=False)
        top10_auc = list(top_auc['epoch'][:top_n_model])
        # print(top10_auc)

        top_loss = self.df.sort_values(by='val_loss')
        top10_loss = list(top_loss['epoch'][:top_n_model])
        # print(top10_loss)

        top10 = list(set(top10_auc + top10_loss))
        # print(top10)

        for epoch in self.df['epoch']:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint_{}_fold{}.pth'.format(epoch, self.fold_idx))
            # print(filename)
            if os.path.isfile(filename):
                if epoch not in top10:
                    os.remove(filename)

    def _save_checkpoint(self, epoch, save_best=False, is_semi=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        if self.ema is not None:
            state_dict = self.ema.ema.state_dict()
        else:
            state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            # 'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        if self.only_save_best:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint_fold{}.pth'.format(self.fold_idx))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint_{}_fold{}.pth'.format(epoch, self.fold_idx))
            # self._save_only_top_10()
        if is_semi:
            filename = os.path.join(self.checkpoint_dir, 'pseudo_checkpoint_fold{}.pth'.format(self.fold_idx))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            best_path = os.path.join(self.checkpoint_dir, 'model_best_fold{}'.format(self.fold_idx))
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        targets = []
        outputs = []
        self.optimizer.zero_grad()
        # for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
        for batch_idx, info in enumerate(tqdm(self.data_loader)):
            data = info['image'].to(self.device)
            target = info['label'].to(self.device)

            with amp.autocast(enabled=self.fp16):

                output = self.model(data, fp16 = self.fp16).squeeze(-1)
                # loss = self.train_criterion(output, target, u_out).mean()
            loss = self.train_criterion(output, target)
                # loss.backward()
                # self.optimizer.step()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            targets.append(target.detach().cpu())
            outputs.append(output.detach().cpu())
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            if batch_idx == self.len_epoch:
                break

            if self.ema:
                self.ema.update(self.model)

        if self.ema:
            self.ema.update_attr(self.model, include=[])

        targets = torch.cat(targets)
        outputs = torch.cat(outputs)
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outputs, targets))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch, self.valid_data_loader)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        csv_log = {'epoch': epoch}
        for key in log:
            csv_log[key] = log[key]
        self.csv_save.update(csv_log)
        return log

    def _valid_epoch(self, epoch, valid_data_loader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.ema is not None:
            self.model_eval = self.ema.ema
        else:
            self.model_eval  = self.model

        
        # state_dict_path = '/home/hana/sonnh/Covid19_Cough_Classification/saved/models/13-Covid19-PlainCNNSmall/0803_233415/checkpoint_{}_fold{}.pth'.format(epoch, self.fold_idx)
        # state_dict = torch.load(state_dict_path)['state_dict']
        # self.model_eval.load_state_dict(state_dict)

        self.model_eval.eval()
        self.valid_metrics.reset()
        targets = []
        outputs = []
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tqdm(valid_data_loader)):
            for batch_idx, info in enumerate(tqdm(valid_data_loader)):
                data = info['image'].to(self.device)
                target = info['label'].to(self.device)
            
                output = self.model(data, fp16 = self.fp16).squeeze(-1)
                loss = self.val_criterion(output, target)

                targets.append(target.detach().cpu())
                outputs.append(output.detach().cpu())
                self.writer.set_step((epoch - 1) * len(valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
            targets = torch.cat(targets)
            outputs = torch.cat(outputs)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _semi_train_epoch(self):
        return None