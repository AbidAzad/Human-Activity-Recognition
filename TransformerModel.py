import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

T = 128
D1 = 9
D2 = 12
H = 5
L = 6
num_classes = 6

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Take the Raw Inertial Samples and Embed them in a Higher Dimension
        self.convolution_series = nn.Sequential(
            nn.Conv1d(D1, D2, 1), nn.GELU(),
            nn.Conv1d(D2, D2, 1), nn.GELU(),
            nn.Conv1d(D2, D2, 1), nn.GELU(),
            nn.Conv1d(D2, D2, 1), nn.GELU())
        
        #Generate an Embedding for each position in the input sequence
        self.position_embedding = nn.Parameter(torch.randn(T + 1, 1, D2))
        
        #Transformer Encoder[Comprised of Self Multi-Head Attention and FC Network]
        self.encoder_layer = TransformerEncoderLayer(d_model = D2, nhead = H, dim_feedforward = 2 * D2, dropout = 0.1, activation = "relu")
        
        #Stack together those EncoderLayers to form the TransformerEncoder
        #The TransformerEncoder will use L TransformerEncoderLayer units
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, L)
        
        #Class Token
        self.cls_token = nn.Parameter(torch.zeros((1, D2)), requires_grad=True)
        
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(D2),
            nn.Linear(D2, D2//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D2//4,  num_classes)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    #Run forward pass given X(input data)
    #Shape of X: N x T x D1
    def forward(self, X):
        #First do the higher dimensional embedding
        X = self.convolution_series(X.transpose(1, 2)).permute(2, 0, 1)
        
        #Generate class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, X.shape[1], 1)
        X = torch.cat([cls_token, X])
        
        #Add Positional Embedding
        X += self.position_embedding
        
        #Run through the Transformer Encoder
        target = self.transformer_encoder(X)[0]
        
        #Get Class Probabilities
        target = self.log_softmax(self.classifier_head(target))
        
        return target
        
        
        
        
        
        
        