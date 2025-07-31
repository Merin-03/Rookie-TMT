import os
import torch

UNK = 0
PAD = 1
BATCH_SIZE = 64
EPOCHS = 20
LAYERS = 6
H_NUM = 8
D_MODEL = 256
D_FF = 1024
DROPOUT = 0.1
MAX_LENGTH = 60

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_FILE = os.path.join(BASE_DIR, 'data/cmn.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_FILE = os.path.join(BASE_DIR, 'data/train.txt')
DEV_FILE = os.path.join(BASE_DIR, 'data/test.txt')
SAVE_FILE = os.path.join(BASE_DIR, 'save/model.pt')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
