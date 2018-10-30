import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        drop_prob = 0.5
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                           dropout=drop_prob, batch_first=True)
        # self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        # Embedding Weights as random uniform
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions):
        captions = captions[:,:-1]
        caption_embed = self.embedding(captions)
        x = torch.cat((torch.unsqueeze(features, 1), caption_embed), 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        word_list = []
        softmax = nn.Softmax(0)
        for ii in range(max_len):
            print(inputs.shape)
            hidden, states = self.lstm(inputs, states)
            outputs = self.fc(hidden)
            print(outputs)
            softmax_score = softmax(outputs[0,0,:])
            print(softmax_score)
            predicted = torch.argmax(softmax_score)
            print(predicted)
            word_list.append(predicted.item())
            inputs = self.embedding(predicted)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = torch.unsqueeze(inputs, 0)
        
        return word_list