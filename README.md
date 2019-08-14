# Setup
1. install virtualenv
2. run `pip install -r requirements.txt`

# P1
1. run `python P1.py <train file>`

# P2
1. run `python P2.py <train file> <dev in file> <dev out file>`

# P3
1. run `python P3.py <train in file>`

# P4
1. run `python P3.py <train file> <dev in file> <dev out file>`

# P5
1. for Bert, run `python bert.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"
2. for Bi-LSTM-CRF, run `python bilstm-crf.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"
3. For attention model, run `python transformers.py <train file> <dev in file> <dev out file> <lang>`, lang is "EN" or "ES"