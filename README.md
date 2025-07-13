**BLaDe:** A **B**ERT-Bi**L**STM-**A**ttention Model for Enhanced Language **De**tection

Language detection is the task of automatically identifying the language of a given sentence. This reprository contains source code files in python which are used in the proposed model, named BLaDe. A small description of source code files is mentioned below:

---------------------------------------
**Pre-requisite:**

Python 3.7+

torch

transformers

scikit-learn

numpy

pandas

NLTK

---------------------------------------
**data_loader.py**

Prepares the dataset for training and evaluation.

Handles dataset loading and processing. It prepares BERT-compatible inputs, computes augmented features, and formats batches for training and evaluation.

---------------------------------------
**blade_model.py**

Defines the model architecture combining BERT, BiLSTM, and an attention mechanism. It also integrates augmented features for enhanced language detection.

---------------------------------------
**train_eval.py**

Contains the training and evaluation pipeline. It manages model training, loss calculation, metric tracking, and model checkpointing.

---------------------------------------
**main.py**

The main execution script. It initializes the dataset, model, and training parameters, and starts the training process using train_eval.py.

---------------------------------------
**test.py**

Used for inference on new or test data. It loads a trained model and generates predictions for evaluation.




















