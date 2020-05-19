# Sentence Variational Autoencoder

Materials borrowed from https://github.com/timbmg/: PyTorch re-implementation of [_Generating Sentences from a Continuous Space_](https://arxiv.org/abs/1511.06349) by Bowman et al. 2015.
![Model Architecture](https://github.com/hammad001/Language-Modelling-CSE291-AS2/blob/master/figs/model.png "Model Architecture")

## Environment setup
1. Install [_anaconda_](https://docs.anaconda.com/anaconda/install/linux/)
2. Create a new environment.
```
conda create -n cse291_as2 python=3.6
```
3. Activate the environment.
```
conda activate cse291_as2
```
4. Install requirements.
```
pip install -r requirements.txt
```

## Download dataset
To run the training, please download the Penn Tree Bank data first (download from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)). The code expects to find at least `ptb.train.txt` and `ptb.valid.txt` in the specified data directory. The data can also be donwloaded with the `dowloaddata.sh` script.

## RNN
### Training
RNNs can be trained using the following command. RNN training uses same arguments as can be provided during training VAE. 
```
python3 train_rnn.py
```
You can add the argument `--tensorboard_logging` to the above command to start logging in tensorboard for plots visualization. Please check the "RNN Train Arguments" section for the full list of argumnets.

Training on a CPU takes approx 17 minutes per epoch.

## VAE
### Training 
Training can be executed with the following command:
```
python3 train.py
```
You can add the argument `--tensorboard_logging` to the above command to start logging in tensorboard for plots visualization. Please check the "VAE Train Arguments" section for the full list of argumnets.

Training on a CPU takes approx 20 minutes per epoch.

### Samples from trained VAE
Sentenes have been obtained after sampling from z ~ N(0, I).  

_mr . n who was n't n with his own staff and the n n n n n_  
_in the n of the n of the u . s . companies are n't likely to be reached for comment_  
_when they were n in the n and then they were n a n n_  
_but the company said it will be n by the end of the n n and n n_  
_but the company said that it will be n n of the u . s . economy_  

## RNN Train Arguments

The following arguments are available:

`--data_dir`  The path to the directory where PTB data is stored, and auxiliary data files will be stored.  
`--create_data` If provided, new auxiliary data files will be created form the source data.  
`--max_sequence_length` Specifies the cut off of long sentences.  
`--min_occ` If a word occurs less than "min_occ" times in the corpus, it will be replaced by the <unk> token.  
`--test` If provided, performance will also be measured on the test set.

`-ep`, `--epochs`  
`-bs`, `--batch_size`  
`-lr`, `--learning_rate`

`-eb`, `--embedding_size`  
`-rnn`, `--rnn_type` Either 'rnn' or 'gru' or 'lstm'.  
`-hs`, `--hidden_size`  
`-nl`, `--num_layers`  
`-bi`, `--bidirectional`  
`-ls`, `--latent_size`  
`-wd`, `--word_dropout` Word dropout applied to the input of the Decoder, which means words will be replaced by `<unk>` with a probability of `word_dropout`.  
`-ed`, `--embedding_dropout` Word embedding dropout applied to the input of the Decoder.

`-v`, `--print_every`  
`-tb`, `--tensorboard_logging` If provided, training progress is monitored with tensorboard.  
`-log`, `--logdir` Directory of log files for tensorboard.  
`-bin`,`--save_model_path` Directory where to store model checkpoints.

## VAE Train Arguments

The following arguments are available:

`--data_dir`  The path to the directory where PTB data is stored, and auxiliary data files will be stored.  
`--create_data` If provided, new auxiliary data files will be created form the source data.  
`--max_sequence_length` Specifies the cut off of long sentences.  
`--min_occ` If a word occurs less than "min_occ" times in the corpus, it will be replaced by the <unk> token.  
`--test` If provided, performance will also be measured on the test set.

`-ep`, `--epochs`  
`-bs`, `--batch_size`  
`-lr`, `--learning_rate`

`-eb`, `--embedding_size`  
`-rnn`, `--rnn_type` Either 'rnn' or 'gru' or 'lstm'.  
`-hs`, `--hidden_size`  
`-nl`, `--num_layers`  
`-bi`, `--bidirectional`  
`-ls`, `--latent_size`  
`-wd`, `--word_dropout` Word dropout applied to the input of the Decoder, which means words will be replaced by `<unk>` with a probability of `word_dropout`.  
`-ed`, `--embedding_dropout` Word embedding dropout applied to the input of the Decoder.

`-af`, `--anneal_function` Default is identity. You will need to implement other annealing methods if you would like to use KL annealing.

`-v`, `--print_every`  
`-tb`, `--tensorboard_logging` If provided, training progress is monitored with tensorboard.  
`-log`, `--logdir` Directory of log files for tensorboard.  
`-bin`,`--save_model_path` Directory where to store model checkpoints.

## Tensorboard Tutorial
[_Tutorial_](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
