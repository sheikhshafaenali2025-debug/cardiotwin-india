"""Hybrid static EHR MLP + temporal CNN-GRU fusion classifier."""
import torch
from torch import nn

class HybridRiskModel(nn.Module):
    def __init__(self, static_dim: int, sequence_features: int=6, hidden: int=32):
        super().__init__()
        self.static_encoder=nn.Sequential(nn.Linear(static_dim,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,hidden),nn.ReLU())
        self.temporal=nn.Sequential(nn.Conv1d(sequence_features,32,kernel_size=3,padding=1),nn.ReLU(),nn.Conv1d(32,32,kernel_size=3,padding=1),nn.ReLU())
        self.gru=nn.GRU(32,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(.15),nn.Linear(hidden,1))
    def forward(self, static, sequence):
        s=self.static_encoder(static.float()); x=self.temporal(sequence.float().transpose(1,2)).transpose(1,2); _,h=self.gru(x)
        return self.head(torch.cat([s,h[-1]],dim=1)).squeeze(-1)
