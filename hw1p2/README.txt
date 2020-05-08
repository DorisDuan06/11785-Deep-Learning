All files are placed under ‘11-785-s20-hw1p2’ folder.

Instruction:

run.py is used to train the model, by the command 'python run.py'.

upgrade.py loads the model that has been saved by run.py and changes the learning rate to reach a higher accuracy. The command is 'python upgrade.py'.

After trying and saving lots of models, a majority vote is performed over the predictions of models with highest performance. The command is 'python majority.py'.


Model:

The structure of the model is {affine - relu - batch} x 4 - affine - softmax, with hidden size = [2048, 1024, 512, 256], context k = 15, batch size = 128, epochs = 20, learning rate = 1e-3, weight decay = 0, loss function = cross entropy loss, optimizer = Adam. Hyperparameters I tuned are above except for loss function and optimizer.

The model first loads the train, dev and test data. When converting them to Dataset, each mel-spectrogram is combined with its previous k spectrogram and next k spectrogram with zero padding on the end of each spectrogram. If there are not enough spectrograms in the neighborhood to combine, spectrograms with all zeros will be combined.

Then the model trains for 20 epochs with 1e-3 learning rate. At each epoch, first run on the training data to get training loss, then evaluate the model using development data and get development loss and accuracy, then keep track of the model with the best development accuracy.

Then the best model is saved, along with the optimizer state, the hidden size, and the batch size.

Then, use the best model to predict the test data. First, run the best model on the test data batch by batch, and concatenate the predictions. Then, write the whole prediction to csv file.

After saving the best model, upgrade.py adjusts the learning rate to be 3e-4 and run for 5 epochs, then run 5 epochs with learning rate 1e-3, then 5 epochs with 3e-4, then 5 epochs with 1e-3, then 5 epochs with 3e-4 to reach 63.5%+ accuracy on the development data.