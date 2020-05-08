from classify import *
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    ''' Load the model '''
    checkpoint = torch.load('resnet')
    model_name = checkpoint['model_name']
    loss_name = checkpoint['loss_name']
    batch_size = checkpoint['batch_size']

    if model_name == 'MobileNet':
        model = MobileNet().to(device)
    elif model_name == 'ResNet':
        model = ResNet(feat_dim).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])  # Note: this model is run on multiple GPUs, so each parameter was stored as "module.xxx"

    criterion_label = nn.CrossEntropyLoss().to(device)
    criterion_loss = nn.CrossEntropyLoss().to(device)
    if loss_name == 'CenterLoss':
        criterion_loss = CenterLoss(2300, feat_dim, device)

    optimizer_label = optim.Adam(model.parameters(), lr=3e-4)
    optimizer_loss = optim.SGD(criterion_loss.parameters(), lr=0.5)
    optimizer_label.load_state_dict(checkpoint['optimizer_label_state_dict'])
    optimizer_loss.load_state_dict(checkpoint['optimizer_loss_state_dict'])
    for g in optimizer_label.param_groups:  # learning rate starts from 3e-4
        g['lr'] = 3e-4

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, patience=3, verbose=True, threshold=1e-2, factor=0.5)

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

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    ''' Retrain the model '''
    epochs = 10
    best_model = model
    best_accuracy = checkpoint['best_accuracy']

    for e in range(epochs):
        start = time.time()
        train_loss = train(model, train_loader, criterion_label, criterion_loss, optimizer_label, optimizer_loss)
        end = time.time()
        val_loss, val_acc = evaluation(model, val_loader, criterion_label, criterion_loss)
        if val_acc > best_accuracy:
            best_model = model
            best_accuracy = val_acc
        scheduler.step(val_loss)
        print("Epoch", e, end - start, "s, Train Loss:", train_loss, ", Val Loss:", val_loss, ", Val Acc:", val_acc)

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
            }, 'UpgradeResNet')

    ''' Model prediction '''
    predict = predict(best_model, test_loader)

    with open('p_classify_Upgrade.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Category"])
        for i, c in enumerate(predict):
            file.writerow([test_dataset.imgs[i][0].split('/')[-1], inv_mapping[c]])
