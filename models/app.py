"""APP Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class APP(nn.Module):
    def __init__(self):
        super(APP, self).__init__()
        self.in_channel = self.out_channel = 128
        
        self.maxpool = nn.MaxPool1d(32, stride=32)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.proj_dim = 72
        
        self.map = nn.Conv1d(64, self.proj_dim, 1, bias=False)
        self.proto_map = nn.Linear(self.in_channel, self.out_channel)
        
        self.reweight = nn.Linear(self.in_channel, 1, bias=False)
        self.reweight_s = nn.Linear(self.in_channel, 1, bias=False)

        self.fc = nn.Linear(self.in_channel, self.out_channel, bias=False)
        self.fc_qs = nn.Linear(self.in_channel, self.out_channel, bias=False)
        self.layer_norm_qs = nn.LayerNorm(self.in_channel) 

    def forward(self, query, supports, prototype):
        # print("?????", query.shape, supports.shape, prototype.shape)
        # torch.Size([2, 2048, 128]) torch.Size([2, 1, 2048, 128]) torch.Size([2, 3, 128])
        
        nway, kshot, PN, dim = supports.shape
        batch = query.shape[0]
        way = nway + 1
        residual = prototype
        
        supports = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0).reshape(-1, PN, dim) # [3, 2048, 128]
        query = self.maxpool(query.transpose(1, 2)).transpose(1, 2) #[2, 64, 128]
        supports = self.maxpool(supports.transpose(1, 2)).transpose(1, 2) #[3, 64, 128]
        supports = supports.reshape(way, kshot, -1, dim)
        
        proto = 0
        for i in range(kshot): 
            support = supports[:, i, :, :] #[3, 64, 128]
            que = self.map(query) #[2, 72, 128]
            sup = self.map(support) #[3, 72, 128]
            new_proto = self.proto_map(prototype) #[2, 3, 128]
            
            # self-correlation to adjust prototype
            que_G, sup_G = que.transpose(1,2) @ que, sup.transpose(1,2) @ sup ##[2, 128, 128] #[3, 128, 128]

            selfcor_q = self.reweight(que_G.unsqueeze(1)).squeeze(-1) / (128. ** 0.5) #[2, 3, 128]
            proto_self_q = torch.sigmoid(selfcor_q) * new_proto #[2, 3, 128]

            selfcor_s = self.reweight_s(sup_G.unsqueeze(0)).squeeze() / (128. ** 0.5) #[2, 3, 128]
            proto_self_s = torch.sigmoid(selfcor_s) * new_proto #[2, 3, 128]

            proto_self = self.fc_qs(proto_self_s) + self.fc_qs(proto_self_q) # [2, 3, 128]
            proto_self = self.layer_norm_qs(proto_self) # [2, 3, 128]
            
            # cross-correlation to adjust prototype
            que, sup = que.reshape(self.proj_dim, -1), sup.reshape(self.proj_dim, -1) #[72, 256] [72, 384]
            crosscor = torch.matmul(que.transpose(0, 1) / (128. ** 0.5), sup) #[256, 384]
            crosscor = crosscor.reshape(batch, dim, way, dim).permute(0, 2, 1, 3) # [2, 3, 128, 128]
            crosscor = F.softmax(crosscor, dim=-1) # [2, 3, 128, 128]
            proto_cross = torch.matmul(crosscor, new_proto.unsqueeze(2).transpose(-2, -1)).squeeze(-1) # [2, 3, 128]
            
            # integrate prototype and adjusted prototypes
            output = self.fc(proto_cross + proto_self) # [2, 3, 128]
            output = self.layer_norm(output + residual) # [2, 3, 128]
            proto = proto + output / kshot # [2, 3, 128]
            
        return proto