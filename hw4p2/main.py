import csv
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from ctcdecode import CTCBeamDecoder
from model import Seq2Seq
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
batch_size = 64
num_workers = 32
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


def Levenshtein(s1, s2):
    N, M = len(s1), len(s2)
    a1, a2 = list(range(M + 1)), [0] * (M + 1)

    for i in range(1, N+1):
        for j in range(M+1):
            if j == 0:
                a2[j] = i
            else:
                a2[j] = min(a2[j-1] + 1,
                            a1[j] + 1,
                            a1[j-1] + int(s1[i-1] != s2[j-1]))
        a1 = a2[:]
    return a1[-1]


def train(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total = 0
    for x, y, x_lens, y_lens in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        x_lens = x_lens.to(device)
        y_lens = y_lens.to(device)

        out, out_lens = model.encoder(x, x_lens)
        loss = criterion(out, y, out_lens, y_lens)
        running_loss += loss.item()
        total += x.size(0)

        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader)
    return running_loss


def evaluation(model, val_loader, criterion):
    model.eval()

    running_loss, running_edit_distance = 0.0, 0.0
    total = 0
    for x, y, x_lens, y_lens in val_loader:
        x = x.to(device)
        y = y.to(device)
        x_lens = x_lens.to(device)
        y_lens = y_lens.to(device)

        out, out_lens = model(x, x_lens)
        loss = criterion(out, y, out_lens, y_lens)
        running_loss += loss.item()
        total += x.size(0)

        # y_pred, _, _, y_pred_lens = decoder.decode(out.transpose(0, 1), out_lens)
        # for i in range(x.shape[0]):
        #     pred = y_pred[i, 0, :y_pred_lens[i, 0]]
        #     pred_label = "".join(PHONEME_MAP[p] for p in pred)
        #     true_label = "".join(PHONEME_MAP[p] for p in y[i, :y_lens[i]])
        #     running_edit_distance += Levenshtein(pred_label, true_label)

    running_loss /= len(val_loader)
    running_edit_distance /= total
    return running_loss, running_edit_distance


def predict(model, test_loader):
    model.eval()

    predict = []
    for x, x_lens in test_loader:
        x = x.to(device)
        x_lens = x_lens.to(device)
        out, out_lens = model(x, x_lens)
        # y, _, _, y_lens = decoder.decode(out.transpose(0, 1), out_lens)
        # for i in range(x.shape[0]):
        #     pred = y[i, 0, :y_lens[i, 0]]
        #     label = "".join(PHONEME_MAP[p] for p in pred if p != 46)
        #     predict.append(label)
    return predict


if __name__ == '__main__':
    ''' Load the data '''
    train_data, val_data, test_data, transcript_train, transcript_val = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_val = transform_letter_to_index(transcript_val, LETTER_LIST)

    train_dataset = Speech2TextDataset(train_data, character_text_train)
    val_dataset = Speech2TextDataset(val_data, character_text_val)
    test_dataset = Speech2TextDataset(test_data, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    ''' Build the model '''
    # decoder = CTCBeamDecoder(['$'] * (N_PHONEMES + 1), beam_width=10, blank_id=46, log_probs_input=True)
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128, isAttended=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction=None)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=1e-2, factor=0.5)

    ''' Train model '''
    epochs = 25
    best_model = None
    best_loss = sys.maxsize

    for e in range(epochs):
        start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer)
        end = time.time()
        val_loss, val_edit_distance = evaluation(model, val_loader, criterion)
        if val_loss < best_loss:
            best_model = model
            best_loss = val_loss
        scheduler.step(val_loss)
        print("Epoch", e + 1, end - start, "s, Train Loss:", train_loss, ", Val Loss:", val_loss, ", Val Edit Distance:", val_edit_distance)

    ''' Save the best model '''
    torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'hidden_size': hidden_size,
            # 'n_layers': n_layers,
            'batch_size': batch_size,
            'best_loss': best_loss
            }, 'lstm_256_3linear_dp')

    ''' Model prediction '''
    pred = predict(best_model, test_loader)

    with open('predict.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Predicted"])
        for i, label in enumerate(pred):
            file.writerow([i, label])
