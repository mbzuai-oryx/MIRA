from llava.train.train import train
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train()
