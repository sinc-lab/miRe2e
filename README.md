# miRe2e

This package contains the original methods proposed in:

    [1] J. Raad, L. A. Bugnon, D. H. Milone and G. Stegmayer, "miRe2e: a full
    end-to-end deep model based on  Transformers for prediction
    of pre-miRNAs from raw genome-wide data", 2021.

miRe2e is a novel deep learning model based on Transformers that allows
finding  pre-miRNA sequences in raw genome-wide data. This model is a full
end-to-end neural architecture, using only the raw sequences as inputs.
This way, there is no need to use other libraries for preprocessing RNA sequences.

The model has 3 stages, as depicted in the figure:

1. Structure prediction model: predicts RNA secondary structure using only 
    the input  sequence.
2. MFE estimation model: estimates the minimum free energy when folding (MFE) the secondary  structure.
3. Pre-miRNA classifier: uses the input RNA sequence and the outputs of the two previous
  models to give a score to the input sequence in order to determine if it is a  pre-miRNA candidate.  
 
![Abstract](abstract.png)

This repository provides a miRe2e model pre-trained with known pre-miRNAs
from *H. sapiens*. It is open sourced and free to use. If you use any of
the following, please cite them properly. 

An easy to try online demo is available at [https://sinc.unl.edu.
ar/web-demo/miRe2e/](https://sinc.unl.edu.ar/web-demo/miRe2e/). This demo runs 
a pre-trained model on small RNA sequences. To use larger datasets, or 
train your oun model, see the following instructions.  

## Installation

You need a Python>=3.7 distribution to use this package. You can install
the package from PyPI:

    pip install miRe2e

depending on your system configuration. You can also clone this repository and install with:

    git clone git@github.com:sinc-lab/miRe2e.git
    cd miRe2e
    pip install .

## Using the trained models

When using miRe2e, pre-trained weights will be automatically downloaded.
The model receives a fasta file with a raw RNA sequence. The sequence is
analyzed with a sliding window, and a pre-miRNA score is assigned to each part. 

You can find a complete demonstration of usage in
[miRe2e usage](https://colab.research.google.com/drive/1xeOrjaYP150War9R-LsPpukpJ7_TV0sh#scrollTo=Uan5dhSzegA2).

The notebook is also in this repository: [miRe2e_usage.ipynb](miRe2e_usage.ipynb).

## Training the models

Training the models may take several hours and requires GPU processing 
capabilities beyond the ones provided freely by google colab.  In  the 
following, there are instructions for training each stage of this 
model. 

Each one of the following steps will train a stage of the model, replacing 
the current model during the rest of the program. New models are saved as 
pickle files (*.pkl). These files can be loaded using 

```python
from miRe2e import MiRe2e
new_model = MiRe2E(mfe_model_file="trained_mfe_predictor.pkl",
                   structure_model_file="trained_structure_predictor.pkl",
                   predictor_model_file="trained_predictor.pkl")
```

  
### Pre-miRNA classifier model

To train the pre-miRNA classifier model, you need at least one set of 
positive samples (known pre-miRNA sequences) and a set of negative samples. 
Each sample must be a trimmed to 100 nt in length to use the current 
model configuration. These should be stored in a single FASTA file, one sample 
per row.  Furthermore, since the pre-miRNAs have an average length of less 
than 100nt, it is  necessary to randomly trim negative training sequences 
to match the positive distribution.  This prevents that training got  
biased by 
the length of the sequences.          

To train this stage, run:

```python
from miRe2e import MiRe2e
model = MiRe2e(device="cuda")
model.fit(pos_fname="positive_examples.fa", 
          neg_fname="negative_examples.fa")
```

### Structure prediction model

To train the Structure prediction model, run:
```python
from miRe2e import MiRe2e
model = MiRe2e(device="cuda")
model.fit_structure("structure_examples.fa")
```

### MFE estimation model

To train the Structure prediction model, run:
```python
from miRe2e import MiRe2e
model = MiRe2e(device="cuda")
model.fit_mfe("mfe_examples.fa")
```
