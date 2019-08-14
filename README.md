# Setup
1. install virtualenv
2. run `pip install -r requirements.txt`

# Instructions to run
- P1: run `python P1.py <train file>`
- P2: run `python P2.py <train file> <dev in file> <dev out file>`
- P3: run `python P3.py <train in file>`
- P4: run `python P3.py <train file> <dev in file> <dev out file>`
- P5 Bert: run `python bert.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"
- P5 Bi-LSTM-CRF: run `python bilstm-crf.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"
- P5 attention model: run `python transformers.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"

# Results:
Using BiLSTM-CRF: 

| Metrics                | EN     | ES     |
|------------------------|--------|--------|
| # Entity in gold data  | 210    | 235    |
| # Entity in prediction | 158    | 156    |
| # Correct Entity       | 118    | 125    |
| Entity precision       | 0.7468 | 0.8013 |
| Entity recall          | 0.5619 | 0.5319 |
| Entity F               | 0.6413 | 0.6394 |
| # Correct sentiment    | 84     | 104    |
| Sentiment precision    | 0.5316 | 0.6667 |
| Sentiment recall       | 0.4000 | 0.4426 |
| Sentiment F            | 0.4565 | 0.5320 |
