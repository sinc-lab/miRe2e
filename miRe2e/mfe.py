import torch as tr
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm

from .encoder import Encoder
from .data_process import load_seq_struct_mfe
from .preprocessor import Preprocessor
from .aux import get_pe


class MFE(nn.Module):
    """Model for RNA MFE estimation."""
    def __init__(self, device="cpu"):
        super(MFE, self).__init__()

        self.device = device
        self.conv1 = Encoder(5, 64, 1, 3)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2first = nn.Linear(12 * 64, 100)
        self.first2med = nn.Linear(100, 30)
        self.FF1 = nn.Linear(30, 1)

        self.m1 = nn.ELU()
        self.m2 = nn.ELU()
        self.m3 = nn.ELU()

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(30)
        self.to(device)

    def forward(self, sentence, structure):
        batch_size = sentence.size(0)
        sentence = tr.cat([sentence, structure], 2)
        output = self.conv1(sentence.permute(0, 2, 1))

        tag_space1 = self.m1(self.hidden2first(output.view(batch_size, -1)))
        tag_space3 = self.m2(self.first2med(self.bn1(tag_space1)))
        tag_scores = self.m3(self.FF1(self.bn2(tag_space3)))
        return tag_scores

    def fit(self, input_fasta, structure_model, batch_size=512,
            max_epochs=200, verbose=True, length=100):

        seq_fasta, _, mfe_fasta = load_seq_struct_mfe(input_fasta)
        structure_model._eval()

        ind = np.arange(len(seq_fasta))
        np.random.shuffle(ind)
        L = int(len(ind) * .8)
        train_ind = ind[:L]
        valid_ind = ind[L:]

        sampler_train = list(
            tr.utils.data.BatchSampler(
                tr.utils.data.RandomSampler(range(len(train_ind)),
                                            replacement=True), batch_size,
                drop_last=True))

        sampler_test = list(tr.utils.data.BatchSampler(
            tr.utils.data.SequentialSampler(range(len(valid_ind))), batch_size,
            drop_last=True))

        optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9,
                              nesterov=True)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 5e-3,
                                                step_size_up=len(
                                                    sampler_train) * 2,
                                                mode='exp_range')

        loss_function = nn.MSELoss()

        Prep = Preprocessor(0)

        # Start training
        loss_min = 999
        # Cache
        struct_e2e = tr.zeros(len(train_ind), length, 1).to(self.device)
        struct_e2e_t = tr.zeros(len(valid_ind), length, 1).to(self.device)

        for epoch in range(max_epochs):
            loss_list = []

            if verbose:
                print("Epoch", epoch)
                train_list = tqdm(sampler_train)
            else:
                train_list = sampler_train


            for num, i in enumerate(train_list):

                # Generate batch
                mfe = tr.zeros(batch_size, 1).to(self.device)
                seq = tr.zeros(batch_size, length, 4).to(self.device)
                largo_seq = tr.zeros(batch_size)

                for j, k in enumerate(i):
                    seq[j, 0:(seq_fasta[train_ind[i[j]]]).size(0), :] = Prep(
                        seq_fasta[train_ind[i[j]]])
                    largo_seq[j] = seq_fasta[train_ind[i[j]]].size(0)

                    mfe[j, :] = mfe_fasta[train_ind[i[j]]]

                self.zero_grad()

                with tr.no_grad():
                    if epoch == 0:
                        PE_batch = get_pe(largo_seq.int(),
                                          length).float().to(self.device)
                        seq_e2e = structure_model(seq, PE_batch)
                        struct_e2e[
                        num * batch_size: num * batch_size + batch_size, :,
                        :] = seq_e2e
                    else:
                        seq_e2e = struct_e2e[
                                  num * batch_size: num * batch_size + batch_size,
                                  :, :]

                prediction = self(seq, seq_e2e)

                loss = loss_function(prediction.view(batch_size, -1).to(self.device),
                                     mfe)

                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()

            loss_tot = np.mean(loss_list)

            self.eval()
            valid_list = []

            with tr.no_grad():

                for num, i in enumerate(sampler_test):

                    # Generate batch
                    mfe = tr.zeros(batch_size, 1).to(self.device)
                    seq = tr.zeros(batch_size, length, 4).to(self.device)
                    largo_seq = tr.zeros(batch_size)

                    for j, k in enumerate(i):
                        seq[j, 0:(seq_fasta[valid_ind[i[j]]]).size(0),
                        :] = Prep(
                            seq_fasta[valid_ind[i[j]]])
                        largo_seq[j] = seq_fasta[valid_ind[i[j]]].size(0)

                        mfe[j, :] = mfe_fasta[valid_ind[i[j]]]

                    if epoch == 0:
                        PE_batch = get_pe(largo_seq.int(),
                                          length).float().to(self.device)
                        seq_e2e = structure_model(seq, PE_batch)
                        struct_e2e_t[
                        num * batch_size: num * batch_size + batch_size, :,
                        :] = seq_e2e
                    else:
                        seq_e2e = struct_e2e_t[
                                  num * batch_size: num * batch_size + batch_size,
                                  :, :]

                    prediction = self(seq, seq_e2e)

                    loss = loss_function(prediction, mfe)
                    if not np.isnan(loss.cpu()):
                        valid_list.append(loss.item())

            loss_valid = np.mean(valid_list)

            if verbose:
                print(f"Train MSE {loss_tot: .4f}, "
                      f"Validation MSE {loss_valid:.4f}")

            self.train()

            # early stop
            if loss_valid < loss_min:
                loss_min = loss_valid
                best_epoch = epoch
                epoch_max = 0
                tr.save(self.state_dict(), 'trained_mfe_predictor.pkl')
            else:
                epoch_max += 1

            if epoch_max >= 30:
                # Recover the best model so far
                self.load_state_dict(tr.load(
                    "trained_mfe_predictor.pkl"))
                if verbose:
                    print(f"Best epoch {best_epoch}: Valid MSE {loss_min}")
                break
