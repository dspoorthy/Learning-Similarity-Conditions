import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.num_conditions = n_conditions
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)
    def forward(self, x, c):
        embedded_x = self.embeddingnet(x)
        mask = self.masks(c)
        if self.learnedmask:
            mask = torch.nn.functional.relu(mask)
        masked_embedding = embedded_x * mask
        masked_embedding = F.normalize(masked_embedding, dim=1) 
        #norm = torch.norm(masked_embedding, p=2, dim=1) + 1e-10
        #masked_embedding = masked_embedding / norm.unsqueeze(-1).expand_as(masked_embedding)
        #return masked_embedding, mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)

        return masked_embedding, torch.norm(mask, dim=1), torch.norm(embedded_x, dim=1), torch.norm(masked_embedding, dim=1) 