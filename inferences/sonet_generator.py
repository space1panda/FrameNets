from pathlib import Path
import sys
cdir = str(Path(__file__).resolve().parents[1])
if cdir not in sys.path:
    sys.path.append(cdir)
import torch
from datasources.worddatasource import WordDatasource
from models.wordmodeltorch import WordModel
import numpy as np
import json

train_params = json.load(open('/home/yegor/Desktop/projects/MLProjects/experiments/sonet.json'))

datas = WordDatasource(path=train_params['path'], seq_len=train_params['seq_len'],
                       max_count=train_params['max_count'])

model = WordModel(embedding_dim=train_params['embedding_dim'],
                  hidden_dim=train_params['hidden_dim'],
                  vocab_size=len(datas._token2ix), bi=train_params["bi"],
                  inference=True)

def get_sample(model, seed, num_lines):
    parameters = torch.load(train_params['save_path'])
    model.load_state_dict(parameters['model_state'])
    model.eval()
    x = 1
    ixs = []
    idx = seed
    count = 0
    while ixs.count(1) < num_lines:
        with torch.no_grad():
            x = torch.tensor(x).view(1, -1)
            out = model(x)
            out = out.numpy()
        idx = np.random.choice(list(range(out.size)), p=out.ravel())
        ixs.append(idx)
        x = idx

    print(' '.join(datas._ix2token[ix] for ix in ixs))

get_sample(model=model, **train_params['inference_params'])