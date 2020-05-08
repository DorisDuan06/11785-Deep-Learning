from classify import *


if __name__ == '__main__':
    ''' Load the data '''
    train_x = np.load('wsj0_train.npy', allow_pickle=True)
    train_y = np.load('wsj0_train_merged_labels.npy', allow_pickle=True)
    val_x = np.load('wsj0_dev.npy', allow_pickle=True)
    val_y = np.load('wsj0_dev_merged_labels.npy', allow_pickle=True)
    test_x = np.load('wsj0_test.npy', allow_pickle=True)

    train_dataset = Phoneme(train_x, train_y)
    val_dataset = Phoneme(val_x, val_y)
    test_dataset = Phoneme(test_x, None, True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    ''' Load the model '''
    checkpoint = torch.load('lstm_256_3linear_dp')
    hidden_size = checkpoint['hidden_size']
    n_layers = checkpoint['n_layers']
    batch_size = checkpoint['batch_size']
    model = RNN(40, N_PHONEMES + 1, hidden_size, n_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    decoder = CTCBeamDecoder(['$'] * (N_PHONEMES + 1), beam_width=20, blank_id=46, log_probs_input=True)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 3e-6

    criterion = nn.CTCLoss(blank=46).to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=1e-2, factor=0.5)

    ''' Train model '''
    epochs = 10
    best_model = model
    best_loss = .35

    for e in range(epochs):
        start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer)
        end = time.time()
        val_loss, val_edit_distance = evaluation(model, decoder, val_loader, criterion)
        if val_loss < best_loss:
            best_model = model
            best_loss = val_loss
        scheduler.step(val_loss)
        print("Epoch", e + 1, end - start, "s, Train Loss:", train_loss, ", Val Loss:", val_loss, ", Val Edit Distance:", val_edit_distance)

    ''' Save the best model '''
    torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'best_loss': val_loss
            }, 'lstm_256_3linear_dp')

    ''' Model prediction '''
    pred = predict(best_model, decoder, test_loader)

    with open('predict_upgrade.csv', "w", newline='') as f:
        file = csv.writer(f, delimiter=',')
        file.writerow(["Id", "Predicted"])
        for i, label in enumerate(pred):
            file.writerow([i, label])
