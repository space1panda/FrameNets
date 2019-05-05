from pathlib import Path
import sys
cdir = str(Path(__file__).resolve().parents[1])
if cdir not in sys.path:
    sys.path.append(cdir)
import pickle
from datasources.chardatasource import CharDatasource
from utils.utils_np import sample, print_sample

def generate_countries(num_countries, path, params_path, seq_len):
    datas = CharDatasource(path=path, seq_len=seq_len)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    for i in range(num_countries):
        sampled_indices = sample(params, datas._char2ix,0)
        print_sample(sampled_indices, datas._ix2char)

# TODO create config file for inference
seq_len = 15
path = '/home/yegor/Desktop/MLProjects/assets/npl_lang_models/test.txt'
params_path = '/home/yegor/Desktop/MLProjects/inferences/fairy_land.pkl'
num_countries = 10

generate_countries(num_countries=num_countries, path=path,
                 params_path=params_path, seq_len=seq_len)

