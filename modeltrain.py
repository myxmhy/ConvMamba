import os.path as osp
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from DataDefine import get_datloader
import torch
from deepspeed.accelerator import get_accelerator
from core import metric, Recorder
from timm.utils import AverageMeter
from utils import (check_dir, print_log, weights_to_cpu,
                   output_namespace)


class modeltrain(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, model_data, model_path):
        """Initialize experiments (non-dist as an example)"""
        self.args = model_data[0]
        self.device = self.args.device
        self.method = model_data[1]
        self.args.method = self.args.method.lower()
        self.epoch = 0
        self.max_epochs = self.args.max_epoch
        self.steps_per_epoch = self.args.steps_per_epoch
        self.rank = self.args.rank
        self.world_size = self.args.world_size
        self.dist = self.args.dist
        self.early_stop = self.args.early_stop_epoch
        self.model_path = model_path
        self.best_loss = 100.

        self.preparation()
        print_log(output_namespace(self.args))
        if self.args.if_display_method_info:
            self.display_method_info()


    def preparation(self):
        """Preparation of basic experiment setups"""
        if self.early_stop <= self.max_epochs // 5:
            self.early_stop = self.max_epochs * 2

        self.checkpoints_path = osp.join(self.model_path, 'checkpoints')
        # load checkpoint
        if self.args.load_from:
            if self.args.load_from == True:
                self.args.load_from = 'latest'
            self.load(name=self.args.load_from)
        # prepare data
        self.get_data()


    def get_data(self):
        """Prepare datasets and dataloaders"""
        (self.args,
        self.train_loader, 
        self.vali_loader, 
        self.test_loader) = get_datloader(self.args)

        folder_path = osp.join(self.model_path, 'std_mean')
        check_dir(folder_path)
        if self.rank == 0:
            np.save(osp.join(folder_path, 'std.npy'), self.args.data_std)
            np.save(osp.join(folder_path, 'mean.npy'), self.args.data_mean)


    def save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self.epoch + 1,
            'optimizer': self.method.optimizer.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()
            }
        if self.rank == 0:
            torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint
    

    def save_checkpoint(self, name=''):
        """Saving models data to checkpoints"""
        checkpoint = weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict())
        if self.rank == 0:
            torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint


    def load(self, name=''):
        """Loading models from the checkpoint"""
        filename = osp.join(self.checkpoints_path, f'{name}.pth')
        try:
            checkpoint = torch.load(filename, map_location='cpu')
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self.load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None and self.args.if_continue:
            self.epoch = checkpoint['epoch']
            self.method.optimizer.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            print_log(f"Successful optimizer state_dict, Lr: {cur_lr:.5e}")
        del checkpoint


    def load_from_state_dict(self, state_dict):
        if self.dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)
        print_log(f"Successful load model state_dict")

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        if self.args.method in ['classification']:
            input_dummy = torch.ones(1, len(self.args.select_channel), self.args.sequence_length).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        print_log('Model info:\n' + info+'\n' + flops+'\n' + dash_line)
        
    def train(self):
        """Training loops of methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self.max_epochs // 10, 30), 
                            rank = self.rank, dist=self.dist, max_epochs = self.max_epochs,
                            method = self.args.method)
        num_updates = self.epoch * self.steps_per_epoch
        vali_loss = False
        early_stop = False
        eta = 1.0  # PredRNN variants
        epoch_time_m = AverageMeter()
        for epoch in range(self.epoch, self.max_epochs):
            begin = time.time()

            if self.dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, train_acc_mean, eta = self.method.train_one_epoch(self.train_loader,
                                                                      epoch, num_updates, eta)

            self.epoch = epoch
            if self.args.val_rate != 0:
                with torch.no_grad():
                    vali_loss, vali_acc = self.vali()
            epoch_time_m.update(time.time() - begin)
            
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            
            epoch_log = 'Epoch: {0}, Steps: {1} | Lr: {2:.5e} | Train Loss: {3:.5e} | Train Acc: {4:.5e}'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, train_acc_mean.avg)
            if vali_loss:
                epoch_log += f" | Vali Loss: {vali_loss:.5e} | Vali Acc: {vali_acc:.5e}"
            print_log(epoch_log)

            print_log(f'Epoch time: {epoch_time_m.val:.2f}s, Average time: {epoch_time_m.avg:.2f}s')

            if self.args.mem_log:
                MemAllocated = round(get_accelerator().memory_allocated() / 1024**3, 2)
                MaxMemAllocated = round(get_accelerator().max_memory_allocated() / 1024**3, 2)
                print_log(f"MemAllocated: {MemAllocated} GB, MaxMemAllocated: {MaxMemAllocated} GB")

            early_stop = recorder(loss_mean.avg, vali_loss, self.method.model, self.model_path, epoch)
            self.best_loss = recorder.val_loss_min

            self.save(name='latest')

            if epoch > self.early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))
                break
            
            if self.args.empty_cache:
                torch.cuda.empty_cache()
            print_log("")
            
        self.save_checkpoint("last_checkpoint")

        if not check_dir(self.model_path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        time.sleep(1)

    def vali(self):
        """A validation loop during training"""
        results, eval_log = self.method.vali_one_epoch(self.vali_loader)
        print_log('Val_metrics\t'+eval_log)

        return results['loss'].mean(), results['acc'].mean()

    def test(self):
        """A testing loop of methods"""
        best_model_path = osp.join(self.checkpoints_path, 'checkpoint.pth')
        self.load_from_state_dict(torch.load(best_model_path))
        
        results = self.method.test_one_epoch(self.test_loader)

        eval_res_av, eval_log_av = metric(results['preds'], results['labels'])
        metrics_list = []
        for i in self.args.metrics:
            metrics_list.append(eval_res_av[i.lower()])
        results['metrics'] = np.array(metrics_list)

        print_log(f"Test: loss: {results['loss'].mean():.5e} \n{eval_log_av}\n")
        folder_path = osp.join(self.model_path, 'saved')
        check_dir(folder_path)
        if self.rank == 0:
            for np_data in ['metrics', 'inputs', 'labels', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])
        return 
