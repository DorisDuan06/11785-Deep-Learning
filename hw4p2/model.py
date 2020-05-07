import torch
import torch.nn as nn
import torch.nn.utils as utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, T, key_size) Key Projection from Encoder per time step
        :param value: (N, T, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        '''
        # print(key.shape)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze()  # (N, T)
        attention = nn.Softmax(1)(energy)  # (N, )
        # print(attention.shape)
        context = torch.bmm(attention, value)
        attention_mask = None
        return context, attention_mask


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
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step,
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if isAttended:
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lens, text=None, isTrain=True, teacher_force_rate=0.9):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[0]

        if isTrain:
            max_len = text.shape[1]
            embeddings = self.embedding(text)  # (N, max_len, hidden_dim)
        else:
            max_len = 250

        predictions = []  # (N, max_len, self.vocab_size)
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, self.vocab_size).to(DEVICE)  # for 1 timestep

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break
            #   out of the loop so you do you do not get index out of range errors.
            if self.isAttended:
                context, _ = self.attention(text[:, i], key, values, lens)
            random_num = torch.rand(1)
            # if
            gumbel_noise = nn.functional.gumbel_softmax()

            if isTrain:
                char_embed = embeddings[:, i, :]  # (N, hidden_dim)
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            inp = torch.cat([char_embed, values[i, :, :]], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]

            prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


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
