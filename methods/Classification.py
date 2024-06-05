import time
import torch
from timm.utils import AverageMeter

from utils import reduce_tensor, get_progress
from .base_method import Base_method
from models import *
from core.lossfun import Regularization

class Classification(Base_method):
        
    def __init__(self, args, ds_config, base_criterion):
        Base_method.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(args)

    def build_model(self, args):
        if args.model == "easytrans":
            return easytrans_model(len(args.select_channel),
                args.num_classes,
                args.nhead,
                args.num_encoder_layers,
                args.num_decoder_layers,
                args.dim_feedforward,
                args.num_conv_layer,
                args.conv_outchannel,
                args.conv_strid_size).to(self.device)
        elif args.model == "easymamba":
            return easymamba_model(len(args.select_channel),
                args.num_classes,
                args.mamba_layers,
                args.d_state,
                args.d_conv,
                args.expand,
                args.num_conv_layer,
                args.conv_outchannel,
                args.conv_strid_size).to(self.device)
        elif args.model == "easypuremamba":
            return easypuremamba_model(len(args.select_channel),
                args.sequence_length,
                args.num_classes,
                args.mamba_layers,
                args.d_state,
                args.d_conv,
                args.expand).to(self.device)
        elif args.model == "easyMHA":
            return easyMHA_model(len(args.select_channel),
                args.num_classes,
                args.MHA_layers,
                args.nhead,
                args.num_conv_layer,
                args.conv_outchannel,
                args.conv_strid_size).to(self.device)
        elif args.model == "easypureMHA":
            return easypureMHA_model(len(args.select_channel),
                args.sequence_length,
                args.num_classes,
                args.MHA_layers,
                args.nhead).to(self.device)
        elif args.model == "easyGRU":
            return easyGRU_model(len(args.select_channel),
                args.num_classes,
                args.hidden_size,
                args.num_layers,
                args.num_conv_layer,
                args.conv_outchannel,
                args.conv_strid_size).to(self.device)
        elif args.model == "easyLSTM":
            return easyLSTM_model(len(args.select_channel),
                args.num_classes,
                args.hidden_size,
                args.num_layers,
                args.num_conv_layer,
                args.conv_outchannel,
                args.conv_strid_size).to(self.device)
        elif args.model == "easypureGRU":
            return easypureGRU_model(len(args.select_channel),
                args.num_classes,
                args.hidden_size,
                args.num_layers).to(self.device)
        elif args.model == "easypureLSTM":
            return easypureLSTM_model(len(args.select_channel),
                args.num_classes,
                args.hidden_size,
                args.num_layers).to(self.device)
        elif args.model == "easypuretrans":
            return easypuretrans_model(len(args.select_channel),
                args.sequence_length,
                args.num_classes,
                args.nhead,
                args.num_encoder_layers,
                args.num_decoder_layers,
                args.dim_feedforward).to(self.device)
    
    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y)
        if self.args.regularization > 0:
            reg_loss = self.reg(self.model)
            loss += reg_loss
        return loss
    
    def predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        pred_y = self.model(batch_x)
        return pred_y
    
    def train_one_epoch(self, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        train_acc_m = AverageMeter()
        self.model.train()
        log_buffer = "Training..."
        progress = get_progress()
        
        if self.args.regularization > 0:
            self.reg=Regularization(self.model, self.args.regularization, p=2, dist=self.dist).to(self.device)

        end = time.time()

        with progress:
            if self.rank == 0:
                train_pbar = progress.add_task(description=log_buffer, total=len(train_loader))
            
            for batch_x, batch_y in train_loader:
                data_time_m.update(time.time() - end)
                if self.by_epoch or not self.dist:
                    self.optimizer.zero_grad()

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.predict(batch_x)
                loss = self.cal_loss(pred_y, batch_y)

                pred = pred_y.argmax(dim=-1)
                train_acc = (pred == batch_y).float().mean()

                if not self.dist:
                    loss.backward()
                else:
                    self.model.backward(loss)

                if self.by_epoch or not self.dist:
                    self.optimizer.step()
                else:
                    self.model.step()
                
                if not self.dist and not self.by_epoch:
                    self.scheduler.step()
                
                torch.cuda.synchronize()
                num_updates += 1

                if self.dist:
                    train_acc_m.update(reduce_tensor(train_acc), batch_x.size(0))
                    losses_m.update(reduce_tensor(loss), batch_x.size(0))
                else:
                    train_acc_m.update(train_acc.item(), batch_x.size(0))
                    losses_m.update(loss.item(), batch_x.size(0))

                if self.rank == 0:
                    log_buffer = 'train loss: {:.4e}'.format(losses_m.avg)
                    log_buffer += ' | train acc: {:.4e}'.format(train_acc_m.avg)
                    log_buffer += ' | data time: {:.4e}'.format(data_time_m.avg)
                    progress.update(train_pbar, advance=1, description=f"{log_buffer}")

                end = time.time()  # end for
        
        if self.by_epoch:
            self.scheduler.step(epoch)

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return num_updates, losses_m, train_acc_m, eta
