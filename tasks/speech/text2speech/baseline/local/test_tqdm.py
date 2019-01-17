import time
import torch

from tqdm import tqdm

def load_data():
    time.sleep(0.001)

def main():
    for i in tqdm(range(1000)):
        load_data()

if __name__ == "__main__":
    main()