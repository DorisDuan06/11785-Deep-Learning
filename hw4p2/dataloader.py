import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def load_data():
    train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
    val = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
    test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('train_transcripts.npy', allow_pickle=True, encoding='bytes')
    transcript_val = np.load('dev_transcripts.npy', allow_pickle=True, encoding='bytes')

    return train, val, test, transcript_train, transcript_val


def transform_letter_to_index(transcript, letter2index):
    letter_to_index_list = []
    for sentence in transcript:
        sentence_to_index = [letter2index['<sos>']]
        for i, word in enumerate(sentence):
            sentence_to_index += [letter2index[l] for l in word.decode()]
            if i < len(sentence) - 1:
                sentence_to_index.append(letter2index[' '])
        sentence_to_index.append(letter2index['<eos>'])
        letter_to_index_list.append(sentence_to_index)
    return letter_to_index_list


def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    for i, letter in enumerate(letter_list):
        letter2index[letter] = i
        index2letter[i] = letter
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if self.isTrain:
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    X = [speech for speech, _ in batch_data]
    Y = [text for _, text in batch_data]
    X_lens = [len(x) for x in X]
    Y_lens = [len(y) for y in Y]

    X = pad_sequence(X, batch_first=True)
    Y = pad_sequence(Y, batch_first=True)
    return X, Y, X_lens, Y_lens


def collate_test(batch_data):
    X = [speech for speech in batch_data]
    X_lens = [len(x) for x in X]
    X = pad_sequence(X, batch_first=True)
    return X, X_lens
