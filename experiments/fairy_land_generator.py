from pathlib import Path
import sys
cdir = str(Path(__file__).resolve().parents[1])
if cdir not in sys.path:
    sys.path.append(cdir)

from trainers.char_trainer_np import CharTrainer

# TODO config file
# TODO terminal parser
n_epochs = 500
hidden = 10
vocab = 27
batch_size = 1
seq_len = 15
path = '/home/yegor/Desktop/MLProjects/assets/npl_lang_models/test.txt'
save_path = '/home/yegor/Desktop/MLProjects/inferences/fairy_land.pkl'
clip_ratio = 5
lr = 0.003


def main():

    trainer = CharTrainer(lr=lr, clip_ratio=clip_ratio,
                     path=path, seq_len=seq_len,
                     batch_size=batch_size,
                     vocab = vocab,
                     hidden=hidden, n_epochs = n_epochs, save_path=save_path)

    trainer.train()

if __name__ == '__main__':
    main()
