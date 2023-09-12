

# Seq2Seq LSTM Model for English to Hindi Transliteration

Implementation of a sequence-to-sequence (seq2seq) model with an LSTM-based encoder-decoder architecture. The purpose of the model is to translate English text to its Hindi transliteration. 





## Datasets
Datasets is a TSV (tab-separated values) file that contains 3 columns: "source", "target", and "source_transliterated". This dataset is a small subset of a larger dataset that contains pairs of Hindi words and their transliterated English versions.
To deploy this project download the Datasets using url

```bash
  url = "https://drive.google.com/file/d/1SbKmisQP63PJnUXgAhXREV0eLbQOy8fm/view"

```
## Model
The model consists of two parts:

TextEncoder: This LSTM-based encoder takes in the English text as input, and returns the hidden and cell states of the LSTM after processing the entire input sequence.
```bash
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hid_size, no_layers, dropout):
        super(TextEncoder, self).__init__()
        self.hid_size = hid_size
        self.no_layers = no_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hid_size, no_layers, dropout=dropout)

    def forward(self, input_seq):
        '''# input_seq shape will be (seq_length, batch_size)'''
        embedded = self.dropout(self.embedding(input_seq))
        # embedded shape will be (seq_length, batch_size, embed_size)
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs shape will be (seq_length, batch_size, hid_size)
        return hidden, cell
```

TextDecoder: This LSTM-based decoder takes in the hidden and cell states from the encoder, and generates the Hindi transliteration character-by-character.

```bash
class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hid_size, output_size, no_layers, dropout):
        super(TextDecoder, self).__init__()
        self.hid_size = hid_size
        self.no_layers = no_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hid_size, no_layers, dropout=dropout)
        self.fc = nn.Linear(hid_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

```
The Seq2SeqModel ties together the encoder and decoder to create the complete model. During training, we use teacher forcing to help stabilize training.

```bash
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(hindi.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs
```

## Installation and Required Libraries

```bash
!pip install torch==1.8.0 torchtext==0.9.0
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import random
from torchtext.legacy.data import Field, BucketIterator,TabularDataset

```

## Documentation

check report [Report link](https://docs.google.com/document/d/17oSvDibdJOfYjOQqfZy1Tog6XcGt71pQrtTVGShOOrA/edit?usp=sharing) for getting other Details.

You can check this 
[Colab link 1](https://colab.research.google.com/drive/1sy5m_BnPzOJhBUTFQCSONIVQNsZki8N5?usp=sharing)
and [Colab link 2](https://colab.research.google.com/drive/108oNTBrGVATYc8LOcGNUe8urREtKfTdT?usp=sharing)
for more info.


