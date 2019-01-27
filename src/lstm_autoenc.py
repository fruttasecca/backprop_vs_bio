import torch
import torch.nn as nn

"""
Lstm auto encoder architecture used in this project, a encoder-decoder architecture
based on lstms. The architecture is as simple as possible given that
the purpose of this project is to test the capabilities of training
"memories" of different algorithms (backprop, or bio-inspired algorithms like cosyne etc.)
"""


class AutoEncoder(nn.Module):
    def __init__(self, device, input_dim, hidden_dim):
        """
        Note: this encoder-decoder class only uses non-bidirectional 1 layered LSTMs.
        1 for encoding and 1 for decoding. :param device: Device to which to map tensors (GPU or CPU).
        :param input_dim: Input size.
        :param hidden_dim Size of the hidden dimension of the recurrent layer.
        """
        super(AutoEncoder, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_dim, self.hidden_dim, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTM(1, self.hidden_dim, batch_first=True, bidirectional=False)

        # to go from hidden dimension size to output size, which is the same as input size
        self.hidden2output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

    def _init_hidden(self, batch_size):
        """
        Init the hidden state.
        :param batch_size: Size of the batch of input.
        :return:
        """
        state = [torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                 torch.zeros(1, batch_size, self.hidden_dim).to(self.device)]
        return state

    def _encoder_forward(self, batch):
        """
        Forward pass of the encoder, given the input batch pass
        it through the encoder and return the final hidden state.
        :param batch: Input batch.
        :return: The final hidden state of the encoder.
        """
        hidden = self._init_hidden(len(batch))

        # forward the whole sequence in the lstm
        _, hidden = self.encoder(batch, hidden)
        return hidden

    def _decoder_forward(self, output_sequence_length, hidden_batch):
        """
        Forward pass of the decoder, starting from the "memory", the hidden
        state of the encoder, output a sequence of the same length of the input.
        :param output_sequence_length: Length of the expected output sequence.
        :param hidden_batch: Batch of hidden states from the encoder.
        :return: Outputs of the same shape of the inputs.
        """
        # forward the whole sequence in the lstm

        # this input has such shape to make sure that the lstm will
        # run/iterate a number of times required to produce a number of outputs equal to the sequence length
        # the input is just a sequence of zeroes

        input = torch.ones((hidden_batch[0].size()[1], output_sequence_length, 1), requires_grad=False).to(self.device)

        outputs, _ = self.decoder(input, hidden_batch)
        return outputs

    def forward(self, batch):
        """
        Forward pass, given an input of size(Batch size, sequence length, token dimension i.e. 1 hotted classes) return
        an output with the same shape, note that the last dimension is not 1 hotted, log softmaxed or anything, it is a real
        valued output from a linear layer. This is to allow custom evaluations/ways of computing probabilities
        downstream (i.e. to see this as a classification problem or a regression problem).
        :param batch: Input batch (batch length, seq length, token length).
        :return: Output of the same size of the input,
        """
        # encode sequence to a memory
        memory = self._encoder_forward(batch)

        # from memory to output sequence
        decoder_outputs = self._decoder_forward(batch.size()[1], memory)

        # send output to fc layer(s)
        decoder_outputs = self.hidden2output(decoder_outputs)
        return decoder_outputs

    def custom_flatten_parameters(self):
        """
        Flatten parameters to run faster, usually needed after "filling" the weights
        of the network in non-canonical ways, as in non gradient descent based algorithms in this
        project (cosyne, etc.).
        :return:
        """
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def total_parameters(self):
        """
        Return the total number of trainable parameters.
        :return:
        """
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params
