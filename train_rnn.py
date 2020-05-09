import os
import json
import time
import torch
import logging
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, experiment_name_rnn
from model_rnn import SentenceRNN

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    model = SentenceRNN(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name_rnn(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    def loss_fn(logp, target, length):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)
        
        return NLL_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp = model(batch['input'], batch['length'])

                # loss calculation
                NLL_loss = loss_fn(logp, batch['target'], batch['length'])
                loss = (NLL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['Loss'] = torch.cat((tracker['Loss'], loss.data))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/NLL_Loss"%split.upper(), NLL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    logger.info("%s Batch %04d/%i, Loss %9.4f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.data[0]))

            logger.info("%s Epoch %02d/%i, Mean Loss %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['Loss'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/Loss"%split.upper(), torch.mean(tracker['Loss']), epoch)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("Model saved at %s"%checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs_rnn')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin_rnn')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
