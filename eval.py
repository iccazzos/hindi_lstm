"""
Based on:

    https://github.com/facebookresearch/colorlessgreenRNNs/blob/main/src/language_models/evaluate_test_perplexity.py

which is in turn based on:

    https://github.com/pytorch/examples/blob/main/word_language_model/main.py
"""



import argparse
import torch
import os
import sys
import csv
from tqdm import tqdm

import data
from data import EOS



def eprint(*args, **kwargs):
    """
    Helper function that prints to stderr.  Use it how you would use `print`.
    """
    print(*args, file=sys.stderr, **kwargs)



parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')
# Model parameters.
parser.add_argument('--data', type=str, default='./vocabdata',
                    help='location of the data corpus')
parser.add_argument('--evaldata', type=str, default='./some_hindi_tokenized.txt',
                    help='evaluation data')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        eprint("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if torch.backends.mps.is_available():
    if not args.mps:
        eprint("WARNING: You have mps device, to enable macOS GPU run with --mps.")
        
use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

# This loads our Hindi vocab.  See `data.py`.
vocab = data.load_vocab_with_eos(args.data, EOS)
ntokens = len(vocab["idx2tok"])

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
else:
    eprint("This script doesn't support Transformer models yet.")
    exit()



def get_logits(prefix):
    """
    Run `prefix` through our model, and return the model output (logits).
    These are pre-softmax values.
    """
    with torch.no_grad():  # no tracking history
        hidden = model.init_hidden(1)
        input = torch.tensor([[ 0 ]], dtype=torch.long).to(device)
        for idx in prefix:
            input.fill_(idx)
            output, hidden = model(input, hidden)

        output = output.squeeze()
        return output



# Load the list of sentences from the evaluation data, calculate the token
# surprisals and print to stdout.
with open(args.evaldata, "r") as fi:
    # The prefix for the current sentence, in indices.
    prefix = []

    # Natural logarithm of 2, used later for calculating surprisal.
    loge_2 = torch.log(torch.tensor([2.])).item()

    for line in fi:
        tok = line.strip()

        if tok == "":
            prefix = []
            print()
            continue

        # Look up the index for this token in the vocabulary.  If not found,
        # raise an exception.
        tok_idx = vocab["tok2idx"][tok]

        if len(prefix) == 0:
            # Can't calculate surprisal for the first token.
            print(f"First token: {tok} (index {tok_idx})")

        else:
            # Get the pre-softmax logits for each token given the prefix.
            logits = get_logits(prefix) # Divide this by a temperature, if desired.

            # This calculates the log probability of the current token `tok` with index `tok_idx`.
            #
            #       log p (tok | prefix)
            #     = log (p* (tok | prefix) / sum_tok p* (tok | prefix))
            #     = log (exp logits [idx] / sum_idx exp logits[idx])
            #     = logits [idx] - log-sum-exp logits
            surprisal_basee = logits[tok_idx] - torch.logsumexp(logits, dim=-1)

            # Change of base to convert natural log to log_2.
            # The result is surprisal, i.e., log_2 p (tok | prefix)
            surprisal_base2 = surprisal_basee / loge_2

            print(f"Next token: {tok} (index {tok_idx}), surprisal = {surprisal_base2}")

        # Add the current token index to the prefix.
        prefix.append(tok_idx)

