import torch.nn.functional as F
import torch as tr
from torch import nn
from .encoder import Encoder
from .positional_encoding import PositionalEncoder
from .data_process import load_train_valid_data
from .focal_loss import FocalLoss
from .get_error import get_error
from .aux import get_pe
from torch import optim
from tqdm import tqdm
from .preprocessor import Preprocessor
import numpy as np


class Predictor(nn.Module):
    """Model for pre-miRNA prediction."""
    def __init__(self, device="cpu"):
        super(Predictor, self).__init__()

        self.feature_dim = 64
        self.hidden_size = 200

        self.PE = PositionalEncoder(self.feature_dim, 6, device=device)

        # Convolutional Layers
        self.conv1 = Encoder(5, self.feature_dim, 3, 4)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=4,
            dim_feedforward=self.feature_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=3)

        # Dropout
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2first = nn.Linear(6 * self.feature_dim + 1, 1000)
        self.first2med = nn.Linear(1000, 1000)
        self.FF1 = nn.Linear(1000, 1000)
        self.FF2 = nn.Linear(1000, 2)
        self.med2tag = nn.Linear(5, 2)  # num_classes)

        self.pool = nn.AvgPool1d(0)

        self.m1 = nn.ELU()
        self.m2 = nn.ELU()

        self.bn1 = nn.BatchNorm1d(6 * self.feature_dim + 1)
        self.bn2 = nn.BatchNorm1d(1000)

        self.Soft = nn.Softmax(dim=1)
        self.HS = nn.Hardshrink(lambd=0.45)
        self.T = nn.Threshold(-0.45, -1)

        self.device = device
        self.to(device)

    def forward(self, sentence, structure, mfe):
        batch_size = sentence.size(0)
        sentence = tr.cat([sentence, structure], 2)
        sentence = self.conv1(sentence.permute(0, 2, 1))
        sentence = self.PE(sentence.view(batch_size, -1, self.feature_dim))
        output = self.transformer_encoder(sentence.view(-1, batch_size,
                                                        self.feature_dim))

        tag_space1 = self.hidden2first((self.bn1(tr.cat([output.view(
            batch_size, -1), mfe], 1))))
        tag_space3 = self.first2med(self.dp2(self.bn2(self.m1(tag_space1))))
        tag_scores = self.Soft(self.FF2(F.elu(self.FF1(F.elu(tag_space3)))))
        return tag_scores

    def run_step(self, structure_model, mfe_model, sequences, labels,
                 seq_cache,
                 mfe_cache, epoch, num,
                 i):
        """

        Parameters
        ----------
        seq batch sequences
        labels batch labels
        epoch number of epoch
        i batch step

        Returns
        -------

        """
        # Generate batch
        seq = tr.zeros(self.batch_size, self.length, 4).to(self.device)
        length_seq = tr.zeros(self.batch_size)
        lab = tr.empty(self.batch_size, 1, 2)
        for j, k in enumerate(i):
            seq[j, 0:(sequences[i[j]]).size(0), :] = \
                self.preprocessor(sequences[i[j]]).to(self.device)
            length_seq[j] = sequences[i[j]].size(0)
            lab[j, :, :] = labels[i[j]]

        # Run forward pass.
        with tr.no_grad():

            # Cache structure and MFE to be used in the predictor.
            if epoch == 0:
                PE_batch = get_pe(length_seq.int(),
                                  self.length).float().to(self.device)
                seq_e2e = structure_model(seq, PE_batch)
                seq_cache[
                num * self.batch_size: (num + 1) * self.batch_size, :,
                :] = seq_e2e
                mfe = mfe_model(seq, seq_e2e)
                mfe_cache[
                num * self.batch_size: (num + 1) * self.batch_size,
                :] = mfe
            else:
                seq_e2e = seq_cache[
                          num * self.batch_size: (num + 1) * self.batch_size,
                          :, :]
                mfe = mfe_cache[
                      num * self.batch_size: (num + 1) * self.batch_size,
                      :]

        prediction = self(seq, seq_e2e, mfe)

        return prediction, lab

    def fit(self, structure_model, mfe_model, pos_fname, neg_fname,
            val_pos_fname, val_neg_fname, batch_size, length, max_epochs,
            verbose):

        self.batch_size = batch_size
        self.length = length

        # Load and prepare dataset
        if verbose:
            print("Loading sequences...")
        train_seq, train_labels, valid_seq, valid_labels = \
            load_train_valid_data(pos_fname, neg_fname, val_pos_fname,
                                  val_neg_fname, length=length)

        if verbose:
            print(f"Training sequences {len(train_seq)} ("
                  f"{int(sum(train_labels[:, 0]))} positive)")
            print(f"Validation sequences {len(valid_seq)} "
                  f"({int(sum(valid_labels[:, 0]))} positive)")

        # Train samples list
        sampler_train = list(tr.utils.data.BatchSampler(
            tr.utils.data.RandomSampler(train_labels), batch_size,
            drop_last=True))
        sampler_valid = list(tr.utils.data.BatchSampler(
            tr.utils.data.SequentialSampler(range(len(valid_seq))), batch_size,
            drop_last=True))

        # Define optimizer, training schedule and loss function
        optimizer = optim.SGD(self.parameters(), lr=1e-3,
                              momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 5e-3,
                                                step_size_up=len(
                                                    sampler_train) * 2,
                                                mode='exp_range')
        ce_weights = tr.Tensor([1, 1]).to(self.device)
        loss_function = FocalLoss(alpha=1, gamma=4, logits=True,
                                  coef=ce_weights, device=self.device)

        # Structure and MFE are computed and cached in the first epoch
        structure_model.eval()
        mfe_model.eval()
        self.preprocessor = Preprocessor(0, device=self.device)
        self.structure_cache = tr.zeros(len(train_seq), self.length,
                                        1).to(self.device)
        self.mfe_cache = tr.zeros(len(train_seq), 1).to(self.device)

        self.structure_cache_valid = tr.zeros(len(valid_seq), self.length,
                                              1).to(self.device)
        self.mfe_cache_valid = tr.zeros(len(valid_seq), 1).to(self.device)

        f1_max = 0
        for epoch in range(max_epochs):
            self.train()
            predictions_all = tr.empty(len(sampler_train) * batch_size,
                                       2).to(self.device)
            labels_all = tr.empty(len(sampler_train) * batch_size, 2).to(self.device)
            loss_all = []

            predictions_all_valid = tr.empty(len(sampler_valid) * batch_size,
                                             2).to(self.device)
            labels_all_valid = tr.empty(len(sampler_valid) * batch_size,
                                        2).to(self.device)
            loss_all_valid = []

            if verbose:
                print("Epoch", epoch)
                train_list = tqdm(sampler_train)
            else:
                train_list = sampler_train

            for num, i in enumerate(train_list):
                # Reset gradients
                self.zero_grad()

                prediction, labels = self.run_step(structure_model,
                                                   mfe_model, train_seq,
                                                   train_labels,
                                                   self.structure_cache,
                                                   self.mfe_cache, epoch, num,
                                                   i)

                loss = loss_function(prediction.view(batch_size, -1).to(
                    self.device), labels.view(batch_size, -1).to(self.device))

                predictions_all[num * batch_size: num * batch_size +
                                                  batch_size, :] = \
                    prediction.view(-1, 2).detach()

                labels_all[num * batch_size: num * batch_size + batch_size,
                :] = labels.view(-1, 2).detach()

                loss_all.append(loss.item())
                loss.backward()

                optimizer.step()
                scheduler.step()

            auct, f1t, pret, rect = get_error(
                labels_all[:, 0].cpu(), predictions_all[:, 0].cpu().detach())
            losst = np.mean(loss_all)

            # Toggle model to test mode for validation
            self.eval()

            with tr.no_grad():

                for num, i in enumerate(sampler_valid):

                    prediction, labels = self.run_step(structure_model,
                                                       mfe_model,
                                                       valid_seq,
                                                       valid_labels,
                                                       self.structure_cache_valid,
                                                       self.mfe_cache_valid,
                                                       epoch, num, i)

                    prediction = prediction.detach().to(self.device)
                    labels = labels.detach().to(self.device)

                    predictions_all_valid[
                    num * batch_size: num * batch_size + batch_size,
                    :] = prediction.view(-1, 2)
                    labels_all_valid[
                    num * batch_size: num * batch_size + batch_size,
                    :] = labels.view(-1, 2)
                    loss = loss_function(prediction.view(-1, 2),
                                         labels.view(-1, 2))
                    if not np.isnan(loss.cpu()):
                        loss_all_valid.append(loss.item())

            lossv = np.mean(loss_all_valid)
            aucv, f1v, prev, recv = get_error(labels_all_valid[:, 0].cpu(),
                                              predictions_all_valid[:,
                                              0].cpu())

            if verbose:
                print(f"Train: Loss {losst: .3f} F1 {f1t: .3f} REC "
                      f"{rect: .3f} PRE {pret: .3f}")
                print(f"Valid: Loss {lossv: .3f} F1 {f1v: .3f} REC "
                      f"{recv: .3f} PRE {prev: .3f}")

            # early stop
            if f1v > f1_max:
                f1_max = f1v
                best_epoch = epoch
                epoch_max = 0
                tr.save(self.state_dict(), 'trained_predictor.pkl')
            else:
                epoch_max += 1
            if epoch_max >= 30:
                # Recover the best model so far
                self.load_state_dict(tr.load(
                    "trained_predictor.pkl"))
                if verbose:
                    print(f"Best epoch {best_epoch}: F1 {f1_max}")
                break

        del self.mfe_cache, self.structure_cache, self.mfe_cache_valid, \
            self.structure_cache_valid

