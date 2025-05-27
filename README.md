Hindi LSTM

# How to use

`eval.py` is the main script in this repository.  It takes a model checkpoint
and a text file containing sentences for evaluation, tokenizes the sentences,
calculates and outputs the per-token surprisal for each sentence.

1.  Get the model checkpoint `fortyepochs.pt` and the Morfessor-based tokenizer
    `hindi_morfessor.bin`.  Feel free to ask me for these two files.

2.  Create a virtual environment and install the necessary packages.  If you
    use conda (or mamba or whatever equivalent), you can do this with one line:

    ```
    $ conda create -n myenv --file package-list.txt
    ```

    `package-list.txt` contains all the packages I installed in the environment
    I used to work on this project.  It definitely contains unnecessary
    packages.

3.  Prepare a text file that contains sentences for evaluation.  An example is
    `some_hindi_text_tokenized.txt`.  This file should consists of sentences in
    the form of newline-separated tokens, with an extra newline between every
    pair of sentences.

    This file is prepared by running a Morfessor-based tokenizer on
    `some_hindi_text_pretokenized.txt`, a file consisting of newline-separated,
    untokenized sentences, with the following script:

    ```
	$ morfessor \
        -T some_hindi_text_pretokenized.txt \
        -l hindi_morfessor.bin \
        -o some_hindi_text_tokenized.txt \
        --output-newlines
    ```

4.  Activate the environment and run `eval.py` like this:

    ```
    $ python eval.py \
        --data vocabdata \
        --evaldata some_hindi_text_tokenized.txt \
        --checkpoint fortyepochs.pt \
        --cuda
    ```

    Omit `--cuda` if you aren't using GPUs.
