import csv
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from model import Seq2Seq
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, create_dictionaries
from search import GreedySearch
from Levenshtein import distance


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
batch_size = 300
num_workers = 32
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
letter2index, index2letter = create_dictionaries(LETTER_LIST)


def train(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    for x, y, x_lens, y_lens in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)  # (N, T)
        y_new = y[:, 1:]

        N, T = y.size()
        out, attention_maps = model(x, x_lens, y)
        out_T = out.permute(0, 2, 1)
        loss = criterion(out_T, y_new)

        loss_mask = torch.arange(T-1).unsqueeze(1) < (torch.LongTensor(y_lens) - 1).unsqueeze(0)
        loss_mask = torch.transpose(loss_mask, 0, 1).to(device)
        masked_loss = loss * loss_mask
        norm_loss = torch.sum(masked_loss) / torch.sum(loss_mask)

        running_loss += norm_loss.item()

        norm_loss.backward()
        clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    running_loss /= len(train_loader)
    return running_loss


def evaluation(model, val_loader, criterion):
    model.eval()

    running_loss, running_edit_distance = 0.0, 0.0
    total = 0
    for x, y, x_lens, y_lens in val_loader:
        x = x.to(device)
        y = y.to(device)  # (N, T)
        y_new = y[:, 1:]

        N, T = y.size()
        out, attention_maps = model(x, x_lens, y)
        out_T = out.permute(0, 2, 1)
        loss = criterion(out_T, y_new)

        loss_mask = torch.arange(T-1).unsqueeze(1) < (torch.LongTensor(y_lens) - 1).unsqueeze(0)
        loss_mask = torch.transpose(loss_mask, 0, 1).to(device)
        masked_loss = loss * loss_mask
        norm_loss = torch.sum(masked_loss) / torch.sum(loss_mask)

        running_loss += norm_loss.item()
        total += x.size(0)

        pred = GreedySearch(out.cpu().detach().numpy(), index2letter)
        true = GreedySearch(y.cpu().detach().numpy(), index2letter)
        for b in range(len(pred)):
            total += len(pred[b])
            running_edit_distance += distance(pred[b], true[b])

    running_loss /= len(val_loader)
    running_edit_distance /= total
    return running_loss, running_edit_distance


def predict(model, test_loader, e):
    model.eval()

    predict = []
    for x, x_lens in test_loader:
        x = x.to(device)
        out, attention_maps = model(x, x_lens, isTrain=False)
        pred = GreedySearch(out.cpu().detach().numpy(), index2letter)
        predict += pred

    with open('predict'+str(e)+'.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Predicted"])
        for i, text in enumerate(pred):
            file.writerow([i, text])


if __name__ == '__main__':
    ''' Load the data '''
    train_data, val_data, test_data, transcript_train, transcript_val = load_data()
    character_text_train = transform_letter_to_index(transcript_train, letter2index)
    character_text_val = transform_letter_to_index(transcript_val, letter2index)

    train_dataset = Speech2TextDataset(train_data, character_text_train)
    val_dataset = Speech2TextDataset(val_data, character_text_val)
    test_dataset = Speech2TextDataset(test_data, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    ''' Build the model '''
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), isAttended=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=1e-2, factor=0.5)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ''' Train model '''
    epochs = 25
    best_model = None
    best_loss = sys.maxsize

    for e in range(epochs):
        start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer)
        end = time.time()
        val_loss, val_edit_distance = evaluation(model, val_loader, criterion)
        predict(model, test_loader, e)
        if train_loss < best_loss:
            best_model = model
            best_loss = train_loss
        scheduler.step(train_loss)
        print("Epoch", e + 1, end - start, "s, Train Loss:", train_loss, ", Val Loss:", val_loss, ", Val Edit Distance:", val_edit_distance)

    ''' Save the best model '''
    torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_size': batch_size,
            'best_loss': best_loss
            }, 'LAS_try')

#     ''' Model prediction '''
#     pred = predict(best_model, test_loader)
