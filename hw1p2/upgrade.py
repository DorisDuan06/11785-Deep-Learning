import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
k = 15


class Phoneme(Dataset):
    def __init__(self, X, Y, pred=False):
        self.indexDict = {}
        self.generateDict(X)
        self.pred = pred
        self.X = X
        if self.pred == False:
            self.Y = Y

    def __len__(self):
        return len(self.indexList)

    def __getitem__(self, i):
        if self.indexDict[i][1] < k:
            up_pad = k - self.indexDict[i][1]
            if len(self.X[self.indexDict[i][0]]) - self.indexDict[i][1] - 1 < k:
                down_pad = k - (len(self.X[self.indexDict[i][0]]) - self.indexDict[i][1] - 1)
                x = np.pad(self.X[self.indexDict[i][0]][0:len(self.X[self.indexDict[i][0]])], ((up_pad, down_pad), (0, 1)), 'constant', constant_values=0)
            else:
                x = np.pad(self.X[self.indexDict[i][0]][0:self.indexDict[i][1]+k+1], ((up_pad, 0), (0, 1)), 'constant', constant_values=0)
        else:
            if len(self.X[self.indexDict[i][0]]) - self.indexDict[i][1] - 1 < k:
                down_pad = k - (len(self.X[self.indexDict[i][0]]) - self.indexDict[i][1] - 1)
                x = np.pad(self.X[self.indexDict[i][0]][self.indexDict[i][1]-k:len(self.X[self.indexDict[i][0]])], ((0, down_pad), (0, 1)), 'constant', constant_values=0)
            else:
                x = np.pad(self.X[self.indexDict[i][0]][self.indexDict[i][1]-k:self.indexDict[i][1]+k+1], ((0, 0), (0, 1)), 'constant', constant_values=0)
        x = torch.flatten(torch.from_numpy(x)).float()
        if self.pred == False:
            y = torch.tensor(self.Y[self.indexDict[i][0]][self.indexDict[i][1]], dtype=torch.int64)
            return x, y
        else:
            return x

    def generateDict(self, X):
        v = 0
        for i, x in enumerate(X):
            for j in range(len(x)):
                self.indexDict[v] = (i, j)
                v += 1
            

class MLP(nn.Module):
    def __init__(self, hidden_size, input_size=(2*k+1)*41, num_classes=138):
        super(MLP, self).__init__()
        layers = []
        num_layers = len(hidden_size) + 1
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size[i]))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_size[i]))
            elif i < num_layers - 1:
                layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_size[i]))
            else:
                layers.append(nn.Linear(hidden_size[i-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, criterion, optimizer):
    model.train()
    
    running_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        loss = criterion(scores, y)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    return running_loss


def evaluation(model, val_loader, criterion):
    model.eval()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predict = torch.max(scores.data, 1)
        total += y.size(0)
        correct += (y == predict).sum().item()
        
        loss = criterion(scores, y).detach()
        running_loss += loss.item()

    running_loss /= len(val_loader)
    accuracy = correct / total
    return running_loss, accuracy


if __name__ == '__main__':    
    train_x = np.load('train.npy', allow_pickle=True)
    train_y = np.load('train_labels.npy', allow_pickle=True)
    val_x = np.load('dev.npy', allow_pickle=True)
    val_y = np.load('dev_labels.npy', allow_pickle=True)
    test = np.load('test.npy', allow_pickle=True)
    
    train_dataset = Phoneme(train_x, train_y)
    val_dataset = Phoneme(val_x, val_y)
    test_dataset = Phoneme(test, None, True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

    epochs = 5

    ''' Load and retrain the MLP model '''
    criterion = nn.CrossEntropyLoss()
    
    checkpoint = torch.load('4layerUpgradeModel')
    hidden_size = checkpoint['hidden_size']
    batch_size = checkpoint['batch_size']
    model = MLP(hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_model = None
    best_accuracy = 0.0
    best_epoch = 0

    model.to(device)

    ''' Train MLP '''
    for e in range(epochs):
        start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer)
        end = time.time()
        val_loss, val_acc = evaluation(model, val_loader, criterion)
        if val_acc > best_accuracy:
            best_model = model
            best_accuracy = val_acc
            best_epoch = e
        print("Epoch", e, end - start, "s, Train Loss:", train_loss, "Val Loss:", val_loss, ", Val Acc:", val_acc)
        
    torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            }, '4layerUpgradeModel')
    
    ''' Evaluate MLP '''
    Predict = None
    for i, x in enumerate(test_loader):
        x = x.to(device)
        score = best_model(x)
        _, predict = torch.max(score.data, 1)
        if i == 0:
            Predict = predict
        else:
            Predict = torch.cat((Predict, predict), 0)
    
    predict = Predict.cpu().numpy()

    with open('predict_4U.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["id", "label"])
        for i, c in enumerate(predict):
            file.writerow([i, c])