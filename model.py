import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
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
        
        self.hidden_dim = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
        
    def forward(self, features, captions):
        # word embedding remove <end>
        cap_embedding = self.embed(captions[:,:-1])
        # cap_embedding shape (batch_size, sentence_size ,embed_size)
        # print('cap_embedding shape ', cap_embedding.shape)
        # features (batch_size, embed_size)
        # print(features.shape)
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        # embeddings shape (batch_size, sentence_size, embed_size)
        # print('embeddings shape ', embeddings.shape)
        lstm_out, self.hidden = self.lstm(embeddings)
        # # lstm_out.shape (batch_size, sentence_size, hidden_size)
        # print(lstm_out.shape)
        outputs = self.linear(lstm_out)
        # outputs.shape (batch_size, sentence_size, vocab_size)
        # print(outputs.shape)
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.linear(outputs.squeeze(1))
            _, predicted = outputs.max(1)
            res.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return res
            