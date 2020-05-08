import torch
import torch.nn as nn
import torch.nn.utils as utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze()  # (N, T)
        attention = nn.Softmax(1)(energy)  # (N, T)

        N, T = attention.size()
        attention_mask = torch.arange(T).unsqueeze(1) < torch.LongTensor(lens).unsqueeze(0)  # (T, N)
        attention_mask = torch.transpose(attention_mask, 0, 1).to(device)

        mask_attention = attention * attention_mask
        norm_attention = mask_attention / torch.sum(mask_attention, dim=1, keepdim=True)

        context = torch.bmm(norm_attention.unsqueeze(1), value).squeeze()
        return context, norm_attention


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim*2, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, len = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        batch_size, Length, dim = x.size()
        if Length % 2 == 1:  # delete last timestep
            x = x[:, :-1, :]
        x = x.contiguous().view(batch_size, Length // 2, dim*2)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, len // 2, batch_first=True, enforce_sorted=False)
        outputs, _ = self.blstm(x)
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.pblstm1 = pBLSTM(hidden_dim*2, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim*2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim*2, hidden_dim)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)

        outputs = self.pblstm1(outputs)
        outputs = self.pblstm2(outputs)
        outputs = self.pblstm3(outputs)

        linear_input, _ = utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.value_size = value_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if isAttended:
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, isTrain=True, teacher_force=0.9):
        batch_size = key.shape[0]

        if isTrain:
            max_len = text.shape[1]
            embeddings = self.embedding(text)  # (N, max_len, hidden_dim)
        else:
            max_len = 250

        predictions = []  # (N, max_len, self.vocab_size)
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, self.vocab_size).to(device)  # for 1 timestep
        context = torch.zeros(batch_size, self.value_size).to(device)  # (N, self.value_size)
        attention_maps = []

        for i in range(max_len-1):  # don't predict on '<eos>'
            if isTrain:
                random_num = torch.rand(1)
                if random_num < teacher_force:  # use ground truth
                    char_embed = embeddings[:, i, :]  # (N, hidden_dim)
                else:  # use previous prediction
                    prediction = nn.functional.gumbel_softmax(prediction)
                    char_embed = self.embedding(prediction.argmax(dim=-1))  # greedy search
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            inp = torch.cat([char_embed, context], dim=1) if self.isAttended else char_embed
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]
            if self.isAttended:
                context, attention_mask = self.attention(output, key, values, lens)

            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))
            attention_maps.append(attention_mask)

        return torch.cat(predictions, dim=1), attention_maps


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, 256)
        self.decoder = Decoder(vocab_size, 256, isAttended=True)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        key, value = self.encoder(speech_input, speech_len)
        if isTrain:
            predictions = self.decoder(key, value, speech_len, text_input)
        else:
            predictions = self.decoder(key, value, speech_len, text=None, isTrain=False)
        return predictions
