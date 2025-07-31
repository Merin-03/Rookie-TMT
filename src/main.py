import time

import numpy as np
from torch.autograd import Variable

from src.config import *
from src.models.transformer import make_model
from src.utils import setup_nltk
from src.utils.data_preparation import PrepareData, subsequent_mask
from src.utils.loss import LabelSmoothing, SimpleLossCompute
from src.utils.optimizers import NoamOpt


def run_epoch(data_iter, model, loss_compute, epoch, is_train):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    if is_train:
        model.train()
    else:
        model.eval()

    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1 and is_train:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch} Batch: {i - 1} Loss: {loss / batch.ntokens:.4f} Tokens per Sec: {(tokens.float() / elapsed / 1000.):.2f}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    best_dev_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer),
                               epoch + 1, is_train=True)
        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

        model.eval()
        print('>>>>> Evaluating...')
        with torch.no_grad():
            dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch + 1,
                                 is_train=False)
        print(f'<<<<< Evaluation Loss: {dev_loss:.4f}')

        if dev_loss < best_dev_loss:
            print(f"Saving new best model with loss: {dev_loss:.4f}")
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
        else:
            print(f"Dev loss did not improve from {best_dev_loss:.4f}.")

        print("-" * 30)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(DEVICE)
    for i in range(max_len - 1):
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data).to(DEVICE)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word).to(DEVICE)], dim=1)
    return ys


def evaluate(data, model):
    model.eval()
    with torch.no_grad():
        for i in range(len(data.dev_en)):
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\nEnglish: " + en_sent)

            cn_sent = "".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("Chinese (Truth): " + cn_sent)

            src_np = np.array(data.dev_en[i])
            src = torch.from_numpy(src_np).long().to(DEVICE).unsqueeze(0)
            src_mask = (src != PAD).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])

            translation = []
            for j in range(1, out.size(1)):
                sym_id = out[0, j].item()
                if sym_id != data.cn_word_dict['EOS']:
                    translation.append(data.cn_index_dict[sym_id])
                else:
                    break
            print("Translation: %s" % "".join(translation))


if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: Raw data file '{RAW_DATA_FILE}' not found.")
        print("Please ensure the file exists and you are running the script from the project's root directory.")
        exit()

    print(">>> Loading and preparing data for model...")
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print(f"Source Vocabulary Size: {src_vocab}")
    print(f"Target Vocabulary Size: {tgt_vocab}")
    print("<<< Data loading complete.")

    print(">>> Initializing Transformer model...")
    model = make_model(
        src_vocab,
        tgt_vocab,
        LAYERS,
        D_MODEL,
        D_FF,
        H_NUM,
        DROPOUT,
        device=DEVICE
    ).to(DEVICE)
    print("<<< Model initialized.")

    print(">>> Setting up loss function and optimizer...")
    criterion = LabelSmoothing(tgt_vocab, padding_idx=PAD, smoothing=0.1)
    optimizer = NoamOpt(D_MODEL, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    print("<<< Loss and optimizer set up.")

    print(">>>>>>> Starting training...")
    train_start = time.time()
    train(data, model, criterion, optimizer)
    print(f"<<<<<<< Finished training, cost {time.time() - train_start:.4f} seconds")

    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    if os.path.exists(SAVE_FILE):
        print(">>>>>>> Loading best model for evaluation...")
        model.load_state_dict(torch.load(SAVE_FILE, map_location=DEVICE))
        print("Model loaded. Starting evaluation...")
        evaluate_start = time.time()
        evaluate(data, model)
        print(f"<<<<<<< Finished evaluation, cost {time.time() - evaluate_start:.4f} seconds")
    else:
        print(f"No saved model found at {SAVE_FILE}. Skipping evaluation.")
