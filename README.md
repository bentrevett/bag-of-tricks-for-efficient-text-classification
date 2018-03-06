# Bag of Tricks for Efficient Text Classification

Implementation of [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) in PyTorch using TorchText

Things you can try:

- Use n-grams by setting `N_GRAMS` > 1. Note: this slows down pre-processing.
- Reduce the vocabulary size by setting `VOCAB_MAX_SIZE` or increasing `VOCAB_MIN_FREQ`
- Train on truncated sequences by setting `MAX_LENGTH`
- Change the tokenizer to a built in one, like the spaCy tokenizer, by setting `TOKENIZER = 'spacy'`. Note: this slows down the pre-processing considerably.
