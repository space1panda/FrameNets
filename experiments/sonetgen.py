from pathlib import Path
import sys
cdir = str(Path(__file__).resolve().parents[1])
if cdir not in sys.path:
    sys.path.append(cdir)

from trainers.wordtrainer import WordTrainer
import torch.nn as nn
import json


train_params = json.load(open('sonet.json'))

def main():

    trainer = WordTrainer(criterion=nn.CrossEntropyLoss(), **train_params)
    trainer._train()

if __name__ == '__main__':
    main()