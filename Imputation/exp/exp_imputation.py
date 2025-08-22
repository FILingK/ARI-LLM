from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
'''
exp_imputation is the main implementation file. For different datasets (Abilene/Geant/WS-DREAM), 
the implementations have slight differences and are saved in the corresponding files 
exp_abilene, exp_geant, and exp_wsdream in the current folder. When running, you need to replace exp_imputation with the corresponding file.
'''
class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, pred, true):
        x_loss = torch.mean(100 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        return x_loss


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

        self.pred_len = 0
        self.seq_len = self.args.seq_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            print(f'len_vali:{len(vali_loader)}')
            for i, batch_x in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                B, T, N = batch_x.shape
                mask = torch.zeros((B, T, N)).to(self.device)
                sample_rate = self.args.mask_rate
                for col in range(N):

                    random_indices = torch.randperm(T)[:int(sample_rate * T)]
                    col_mask = torch.zeros(T, device=batch_x.device)
                    col_mask[random_indices] = 1

                    mask[:, :, col] = col_mask.unsqueeze(0).expand(B, -1)

                inp = batch_x.masked_fill(mask == 0, 0)


                for col in range(N):

                    col_mask = mask[0, :, col]

                    unmasked_indices = torch.nonzero(col_mask, as_tuple=True)[0]

                    masked_indices = torch.nonzero(col_mask == 0, as_tuple=True)[0]

                    if len(masked_indices) > 0:

                        random_mapping = torch.randint(0, len(unmasked_indices), (len(masked_indices),))

                        inp[:, masked_indices, col] = batch_x[:, unmasked_indices[random_mapping], col]

                known_rate = self.args.mask_rate * 100
                target_rate = known_rate
                times = np.ceil(np.log2(100 / known_rate)).astype(int)
                outputs = inp
                for j in range(times):
                    known_rate = target_rate
                    target_rate = min(known_rate * 2, 100)
                    outputs = self.model(outputs, None, known_rate, target_rate)
                outputs = outputs.detach()

                pred = outputs.cpu()
                true = batch_x.detach().cpu()
                mask = mask.cpu()
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        scaler = torch.cuda.amp.GradScaler()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = nn.L1Loss()

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f'len_train:{len(train_loader)}')
            for i, batch_x in tqdm(enumerate(train_loader)):
                all_loss = 0
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                B, T, N = batch_x.shape

                inp_list = []

                inp_2 = batch_x.clone()

                for i in range(0, T, 50):
                    inp_2[:, i:i + 49, :] = batch_x[:, i + 49:i + 50, :]
                inp_list.append(inp_2)

                inp_4 = batch_x.clone()

                for i in range(0, T, 25):
                    inp_4[:, i:i + 24, :] = batch_x[:, i + 24:i + 25, :]
                inp_list.append(inp_4)

                inp_8 = batch_x.clone()
                for i in range(0, 48, 12):
                    inp_8[:, i:i + 11, :] = batch_x[:, i + 11:i + 12, :]

                inp_list.append(inp_8)

                inp_16 = batch_x.clone()
                for i in range(0, 48, 6):
                    inp_16[:, i:i + 5, :] = batch_x[:, i + 5:i + 6, :]

                inp_list.append(inp_16)

                inp_32 = batch_x.clone()
                for i in range(0, 48, 3):
                    inp_32[:, i:i + 2, :] = batch_x[:, i + 2:i + 3, :]

                inp_list.append(inp_32)

                inp_64 = batch_x.clone()
                for i in range(0, 48, 3):
                    inp_64[:, i:i + 1, :] = batch_x[:, i + 2:i + 3, :]

                inp_list.append(inp_64)

                inp_100 = batch_x.clone()
                inp_list.append(inp_100)

                target_rate = 2
                for i in range(len(inp_list) - 1):
                    # print(f"step:{i}")
                    known_rate = target_rate
                    target_rate = min(known_rate * 2, 100)
                    outputs = self.model(inp_list[i], None, known_rate, target_rate)
                    loss = criterion(outputs, inp_list[i + 1])
                    all_loss = all_loss + loss

                train_loss.append(all_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, all_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                all_loss.backward()
                model_optim.step()

                torch.cuda.empty_cache()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            torch.cuda.empty_cache()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []

        self.model.eval()
        features = 0
        with torch.no_grad():
            print(f'len_test:{len(test_loader)}')
            for i, batch_x in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                B, T, N = batch_x.shape
                features = N

                mask = torch.zeros((B, T, N)).to(self.device)
                sample_rate = self.args.mask_rate
                for col in range(N):
                    random_indices = torch.randperm(T)[:int(sample_rate * T)]
                    col_mask = torch.zeros(T, device=batch_x.device)
                    col_mask[random_indices] = 1

                    mask[:, :, col] = col_mask.unsqueeze(0).expand(B, -1)

                inp = batch_x.masked_fill(mask == 0, 0)

                for col in range(N):

                    col_mask = mask[0, :, col]

                    unmasked_indices = torch.nonzero(col_mask, as_tuple=True)[0]

                    masked_indices = torch.nonzero(col_mask == 0, as_tuple=True)[0]

                    if len(masked_indices) > 0:
                        random_mapping = torch.randint(0, len(unmasked_indices), (len(masked_indices),))

                        inp[:, masked_indices, col] = batch_x[:, unmasked_indices[random_mapping], col]

                known_rate = self.args.mask_rate * 100
                target_rate = known_rate
                times = np.ceil(np.log2(100 / known_rate)).astype(int)
                outputs = inp
                for j in range(times):
                    known_rate = target_rate
                    target_rate = min(known_rate * 2, 100)
                    outputs = self.model(outputs, None, known_rate, target_rate)
                outputs = outputs.detach()

                outputs = outputs.cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        print('test shape:', preds.shape, trues.shape)

        nmae, nrmse, kl = metric(preds[masks == 0], trues[masks == 0])

        print('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('nmae:{}, nrmse:{} kl:{}'.format(nmae, nrmse, kl))

        f.write('\n')
        f.write('\n')
        f.close()

        return