import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import ResNet, MobileNet
from losses import CenterLoss


cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
num_workers = 32
loss_weight = 3e-3  # feature loss weight
feat_dim = 4096


def train(model, train_loader, criterion_label, criterion_loss, optimizer_label, optimizer_loss):
    model.train()

    running_loss = 0.0
    for x, y in train_loader:
        optimizer_label.zero_grad()
        optimizer_loss.zero_grad()
        x = x.to(device)
        y = y.to(device)

#         if model.__class__ == MobileNet:
#             loss_output, label = model(x)
#         elif model.__class__ == ResNet:
        loss_output, label, features = model(x)
        _, predict = torch.max(label.data, 1)
        l_loss = criterion_label(label, y.long())
        f_loss = criterion_loss(loss_output, y.long()) if criterion_loss.__class__ != nn.CrossEntropyLoss else 0
        loss = l_loss + loss_weight * f_loss

        running_loss += loss.item()

        loss.backward()
        optimizer_label.step()
        for param in criterion_loss.parameters():
            param.grad.data *= (1. / loss_weight)
        optimizer_loss.step()

    running_loss /= len(train_loader)
    return running_loss


def evaluation(model, val_loader, criterion_label, criterion_loss):
    model.eval()

    running_loss = 0.0
    total, correct = 0.0, 0.0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

#         if model.__class__ == MobileNet:
#             loss_output, label = model(x)
#         elif model.__class__ == ResNet:
        loss_output, label, features = model(x)
        _, predict = torch.max(label.data, 1)
        total += y.size(0)
        correct += (y == predict).sum().item()

        l_loss = criterion_label(label, y.long())
        f_loss = criterion_loss(loss_output, y.long()) if criterion_loss.__class__ != nn.CrossEntropyLoss else 0
        loss = l_loss + loss_weight * f_loss

        running_loss += loss.item()

    running_loss /= len(val_loader)
    accuracy = correct / total
    return running_loss, accuracy


def predict(model, test_loader):
    Predict = None
    for i, (x, _) in enumerate(test_loader):
        x = x.to(device)
#         if model.__class__ == MobileNet:
#             _, label = model(x)
#         elif model.__class__ == ResNet:
        _, label, _ = model(x)
        _, predict = torch.max(label.data, 1)
        if i == 0:
            Predict = predict
        else:
            Predict = torch.cat((Predict, predict), 0)

    predict = Predict.cpu().numpy()
    return predict


if __name__ == '__main__':
    model_name = 'ResNet'
    loss_name = 'CenterLoss'

    ''' Load the data '''
    transformations = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    test_transformations = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder('train_data/medium', transform=transformations)
    val_dataset = datasets.ImageFolder('validation_classification/medium', transform=test_transformations)
    test_dataset = datasets.ImageFolder('test_classification', transform=test_transformations)
    inv_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}

    batch_size = 256

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    ''' Build the model '''
    if model_name == 'MobileNet':
        model = MobileNet().to(device)
    elif model_name == 'ResNet':
        model = ResNet(feat_dim).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_label = nn.CrossEntropyLoss().to(device)
    criterion_loss = nn.CrossEntropyLoss().to(device)
    if loss_name == 'CenterLoss':
        criterion_loss = CenterLoss(2300, feat_dim, device)

    optimizer_label = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_loss = optim.SGD(criterion_loss.parameters(), lr=0.5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, patience=3, verbose=True, threshold=1e-2, factor=0.5)

    ''' Train model '''
    epochs = 80 if model_name == 'MobileNet' else 40
    best_model = None
    best_accuracy = 0.0

    for e in range(epochs):
        start = time.time()
        train_loss,  = train(model, train_loader, criterion_label, criterion_loss, optimizer_label, optimizer_loss)
        end = time.time()
        val_loss, val_acc = evaluation(model, val_loader, criterion_label, criterion_loss)
        if val_acc > best_accuracy:
            best_model = model
            best_accuracy = val_acc
        scheduler.step(val_loss)
        print("Epoch", e + 1, end - start, "s, Train Loss:", train_loss, ", Val Loss:", val_loss, ", Val Acc:", val_acc)

    ''' Save the best model '''
    torch.save({
            'model_name': model_name,
            'loss_name': loss_name,
            'model_state_dict': best_model.state_dict(),
            'optimizer_label_state_dict': optimizer_label.state_dict(),
            'optimizer_loss_state_dict': optimizer_loss.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'batch_size': batch_size,
            'best_accuracy': best_accuracy
            }, 'resnet')

    ''' Model prediction '''
    predict = predict(best_model, test_loader)

    with open('p_classify.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Category"])
        for i, c in enumerate(predict):
            file.writerow([test_dataset.imgs[i][0].split('/')[-1], inv_mapping[c]])
