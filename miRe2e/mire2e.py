import numpy as np
import torch as tr
from tqdm import tqdm

from torch.hub import load_state_dict_from_url


from .predictor import Predictor
from .mfe import MFE
from .structure import Structure
from .data_process import data_load
from .preprocessor import Preprocessor
from .aux import gen_code, get_pe


BASE_URL = "https://github.com/sinc-lab/miRe2e/blob/master/models/"
PRETRAINED = {"animals-structure": f"{BASE_URL}structure.pkl?raw=true",
              "animals-mfe": f"{BASE_URL}mfe.pkl?raw=true",
              "animals-predictor": f"{BASE_URL}predictor.pkl?raw=true",
              "hsa-structure": f"{BASE_URL}structure-hsa.pkl?raw=true",
              "hsa-mfe": f"{BASE_URL}mfe-hsa.pkl?raw=true",
              "hsa-predictor": f"{BASE_URL}predictor-hsa.pkl?raw=true"}

class MiRe2e:
    """End-to-end deep learning model based on Transformers for
    pre-miRNA prediction."""
    def __init__(self, device="cpu", pretrained="hsa", mfe_model_file=None,
                 structure_model_file=None, predictor_model_file=None,
                 length=100):
        """
        Initialize a miRe2e instance.

        Parameters
        ----------
        device: Either "cpu" or "cuda"
        pretrained: Use pretrained models: "hsa" for h. sapiens, "animals"
        for a set from several animals, or "no" to avoid using pre-trained
        weights.
        mfe_model_file: Alternative model file for MFE prediction
        structure_model_file: Alternative model file for structure prediction
        predictor_model_file: Alternative model file for pre-miRNA prediction
        length: Sequence window length
        """

        self._predictor = Predictor(device=device)
        self._structure = Structure(device=device)
        self._mfe = MFE(device=device)
        self.length = length

        self.device = device

        self.preprocessor = Preprocessor(0, device)

        if structure_model_file is None:
            if pretrained != "no":
                state_dict = load_state_dict_from_url(
                    PRETRAINED[f"{pretrained}-structure"], map_location=device)
                self._structure.load_state_dict(state_dict)
        else:
            state_dict = tr.load(structure_model_file, map_location=device)
            self._structure.load_state_dict(state_dict)

        if mfe_model_file is None:
            if pretrained != "no":
                state_dict = load_state_dict_from_url(
                    PRETRAINED[f"{pretrained}-mfe"], map_location=device)
                self._mfe.load_state_dict(state_dict)
        else:
            state_dict = tr.load(mfe_model_file, map_location=device)
            self._mfe.load_state_dict(state_dict)


        if predictor_model_file is None:
            if pretrained != "no":
                state_dict = load_state_dict_from_url(
                    PRETRAINED[f"{pretrained}-predictor"], map_location=device)
                self._predictor.load_state_dict(state_dict)
        else:
            state_dict = tr.load(predictor_model_file, map_location=device)
            self._predictor.load_state_dict(state_dict)

    def _eval(self):
        self._structure.eval()
        self._mfe.eval()
        self._predictor.eval()

    def predict(self, filename, length=100, step=20, batch_size=4096,
                verbose=True):
        """Run a pre-miRNA prediction on raw RNA sequence.

        Parameters
        ----------
        filename : str
            Fasta file to predict.
        length : int
            Window length (defined by trained model).
        step : int
            Window step
        batch_size : int
            Batch size for prediction, larger is faster but uses more
            resources.
        verbose : bool
            Print status on console.

        Returns
        -------
        scores_5_3 : array
            Pre-miRNA scores in the 5'-3' direction.
        scores_3_5 : array
            Pre-miRNA scores in the 3'-5' direction.
        index : array
            Sequence names containing positions in base-pairs.
        """
        self._eval()

        if verbose:
            print("Loading sequences...")
        data = data_load(length, step, filename)
        if verbose:
            print(f"Number of sequences: {len(data)}")
            print("Done")

        sampler = list(tr.utils.data.BatchSampler(
            tr.utils.data.SequentialSampler(range(len(data))), batch_size,
                drop_last=False))

        index, scores = [], []

        with tr.no_grad():
            # Analize the raw sequence by windows batches.

            if verbose:
                sampler = tqdm(sampler)

            for num, i in enumerate(sampler):
                # Generate batch
                seq = tr.zeros(len(i), length, 4).to(self.device)
                seq_length = tr.zeros(len(i))
                name = []

                for j, k in enumerate(i):
                    seq[j, :, :] = self.preprocessor(gen_code(data[i[j]].seq))
                    seq_length[j] = length
                    name.append(data[i[j]].id)

                pe_batch = get_pe(seq_length.int(), length).float().to(self.device)

                tag_scores = self.forward(seq, pe_batch).detach().cpu()

                # Save positions
                index += name

                # Save non-padded scores
                # First column is mirna class
                scores.append(tag_scores[:, 0])


        scores = np.concatenate(scores)
        scores_5_3 = scores[:len(scores)//2]
        scores_3_5 = scores[len(scores)//2:]
        index = index[:len(index)//2]

        return scores_5_3, scores_3_5, index

    def forward(self, seq, pe_batch):
        """Predict pre-miRNAs given a sequence batch"""

        seq_e2e = self._structure(seq, pe_batch).detach()
        mfe = self._mfe(seq, seq_e2e).detach()
        tag_scores = self._predictor(seq, seq_e2e, mfe).view(-1, 2).detach()

        return tag_scores

    def fit_structure(self, input_fasta, batch_size=512):
        """Fit structure estimator

        Parameters
        ----------
        input_fasta : str
            A fasta with hairpin-like structures. This file should
        be a list of several sequences. Each line in the file have the
        sequence and expected structure to train the model.
        batch_size : int
            Training batch size

        Returns
        -------

        """
        # Reset current model
        self._structure = Structure(device=self.device)
        self._structure.fit(input_fasta, batch_size=batch_size)

    def fit_mfe(self, input_fasta, batch_size=512):
        """Fit MSE estimator. It uses previously trained structure model.

        Parameters
        ----------
        input_fasta: A fasta with sequence, structures and MFE values. This
        file should be a list of several sequences. It should contain
        examples of flats, hairpin-like and pre-mirnas to be usefull for
        mirna classification.

        Returns
        -------

        """
        # Reset current model
        self._mfe = MFE(device=self.device)
        self._mfe.fit(input_fasta, self._structure, batch_size=batch_size)

    def fit(self, pos_fname: str, neg_fname: str, val_pos_fname=None,
            val_neg_fname=None, verbose=True, batch_size=512,
            max_epochs=150, length=100):
        """
        Fit pre-miRNA predictor. Asumes that Structure and MFE models are
        already trained. Validation sequences are optional. In that case,
        training sequences are splitted in 80-20% scheme.

        It uses previously trained structure and MFE models.

        Parameters
        ----------
        pos_fname: Fasta file with known pre-miRNAs used for training.
        neg_fname: Fasta file with negative samples for training.
        val_pos_fname: Fasta file with known pre-miRNAs used for early-stop.
        val_neg_fname: Fasta file with negative samples for early-stop.

        Returns
        -------

        """
        # Reset current model
        self._predictor = Predictor(device=self.device)

        self._predictor.fit(self._structure, self._mfe, pos_fname, neg_fname,
                            val_pos_fname, val_neg_fname,
                            batch_size=batch_size, length=length,
                            max_epochs=max_epochs, verbose=verbose)
