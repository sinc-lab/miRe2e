import torch as tr
import numpy as np
from tqdm import tqdm

from torch import nn
from torch import optim
from .encoder import EncoderStr
from .aux import get_pe
from .preprocessor import Preprocessor
from .data_process import load_seq_struct_mfe

class Structure(nn.Module):
    """Model for RNA secondary structure prediction."""

    def __init__(self, device='cpu'):
        super(Structure, self).__init__()

        self.feature_dim = 111
        self.hidden_size = 200

        self.conv_in = EncoderStr(4, self.feature_dim, 1, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim * 2, nhead=2,
            dim_feedforward=self.feature_dim * 4 * 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=6)

        # Dropout
        self.dp1 = nn.Dropout(p=0.0)
        self.dp2 = nn.Dropout(p=0.0)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2first = nn.Linear(222, 100)
        self.first2med = nn.Linear(100, 10)
        self.med2tag = nn.Linear(10, 1)  # num_classes)

        self.m1 = nn.ELU()
        self.m2 = nn.ELU()
        self.Soft = nn.LogSoftmax(dim=2)
        self.T = nn.Tanh()

        self.device = device
        self.to(device)

    def forward(self, sentence, pe_batch):
        sentence = self.conv_in(sentence.permute(0, 2, 1))
        sentence = tr.cat([sentence, pe_batch.permute(0, 2, 1)], 1)
        emb = self.transformer_encoder(sentence.permute(2, 0, 1))

        tag_space1 = self.hidden2first(self.dp1(emb.permute(1, 0, 2)))
        tag_space2 = self.first2med(self.dp2((self.m1(tag_space1))))
        tag_scores = self.med2tag((self.m2(tag_space2)))
        return self.T(tag_scores)


    def fit(self, input_fasta, batch_size=512, max_epochs=200,
            length=100, verbose=True):
        """ Train structure prediction model.
        Parameters
        ----------
        input_fasta: fasta file containing the sequence and structure of
        short sequences (less than "length")
        batch_size
        device
        max_epochs
        length

        Returns
        -------

        """

        if verbose:
            print("Loading sequences...")
        sequence, structure, _ = load_seq_struct_mfe(input_fasta)
        if verbose:
            print(f"Done ({len(sequence)} sequences)")

        assert len(sequence)>=10*batch_size, f"batch_size should be between 1 and 1/10 the number of sequences. batch_size={batch_size} was given for {len(sequence)} sequences"

        ind = np.arange(len(sequence))
        np.random.shuffle(ind)
        L = int(len(ind)*.8)
        train_ind = ind[:L]
        valid_ind = ind[L:]

        sampler_train = list(
            tr.utils.data.BatchSampler(
                tr.utils.data.RandomSampler(range(len(train_ind)),
                                            replacement=True), batch_size,
                drop_last=False))

        sampler_test = list(tr.utils.data.BatchSampler(
            tr.utils.data.SequentialSampler(range(len(valid_ind))), batch_size,
            drop_last=False))

        optimizer = optim.SGD(self.parameters(), lr=1)
        scheduler = tr.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        loss_function = nn.MSELoss()

        Prep = Preprocessor(0)
        Prep_str = Preprocessor(1)

        # Start training
        loss_min = 999
        for epoch in range(max_epochs):
            loss_list = []

            if verbose:
                print("Epoch", epoch)
                train_list = tqdm(sampler_train)
            else:
                train_list = sampler_train

            for num, i in enumerate(train_list):

                # Generate batch
                seq = tr.zeros(batch_size, length, 4).to(self.device)
                largo_seq = tr.zeros(batch_size)
                seq_struct = tr.zeros(batch_size, length, 1).to(self.device)

                for j, k in enumerate(i):
                    seq[j, 0:(sequence[train_ind[i[j]]]).size(0), :] = Prep(
                        sequence[train_ind[i[j]]])

                    largo_seq[j] = sequence[train_ind[i[j]]].size(0)
                    seq_struct[j, 0:(structure[train_ind[i[j]]]).size(0),
                    :] = Prep_str(
                        structure[train_ind[i[j]]][0:(structure[train_ind[i[
                            j]]]).size(0)])

                self.zero_grad()

                PE_batch = get_pe(largo_seq.int(), length).float().to(self.device)
                prediction = self(seq, PE_batch)

                loss = loss_function(prediction.view(batch_size, -1),
                                     seq_struct.view(batch_size, -1))

                loss_list.append(loss.item())
                loss.backward()
                tr.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

            scheduler.step()

            loss_tot = np.mean(loss_list)

            self.eval()

            valid_list = []

            with tr.no_grad():

                for num, i in enumerate(sampler_test):

                    # Generate batch
                    seq = tr.zeros(batch_size, length, 4).to(self.device)
                    largo_seq = tr.zeros(batch_size)
                    seq_struct = tr.zeros(batch_size, length, 1).to(self.device)
                    Y = tr.empty(batch_size, 1, 2)
                    for j, k in enumerate(i):
                        seq[j, 0:(sequence[valid_ind[i[j]]]).size(0),
                        :] = Prep(sequence[valid_ind[i[j]]])
                        largo_seq[j] = sequence[valid_ind[i[j]]].size(0)
                        seq_struct[j, 0:(structure[valid_ind[i[j]]]).size(0),
                        :] = Prep_str(
                            structure[valid_ind[i[j]]][0:(structure[
                                valid_ind[i[
                                j]]]).size(0)])


                    PE_batch = get_pe(largo_seq.int(), length).float().to(
                        self.device)

                    prediction = self(seq, PE_batch)

                    loss = loss_function(prediction.view(batch_size, -1),
                                         seq_struct.view(batch_size, -1))

                    if not np.isnan(loss.cpu()):
                        valid_list.append(loss.item())

            loss_valid = np.mean(valid_list)

            if verbose:
                print(f"Train loss {loss_tot: .4f}, "
                      f"Validation loss {loss_valid:.4f}")

            self.train()

            # early stop
            if loss_valid < loss_min:
                loss_min = loss_valid
                best_epoch = epoch
                epoch_max = 0
                tr.save(self.state_dict(), 'trained_structure_predictor.pkl')
            else:
                epoch_max += 1

            if epoch_max >= 30:
                # Recover the best model so far
                self.load_state_dict(tr.load(
                    "trained_structure_predictor.pkl"))
                if verbose:
                    print(f"Best epoch {best_epoch}: Valid Loss {loss_min}")
                break
