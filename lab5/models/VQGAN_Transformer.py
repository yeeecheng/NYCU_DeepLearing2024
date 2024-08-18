import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        # do not used!!
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _  =self.vqgan.encode(x)
        return codebook_mapping, codebook_indices.view(codebook_mapping.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiments, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cube":
            return lambda r: 1 - r ** 2
        elif mode == "sin":
            return lambda r: 1 - np.sin(r * np.pi / 2)
        elif mode == "sqrt":
            return lambda r: 1 - np.sqrt(r)
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        # stage 2 MVTM
        
        #ground truth tokens
        _, visual_tokens = self.encode_to_z(x)
        # create mask
        mask = torch.bernoulli(0.5 * torch.ones(visual_tokens.shape, device=visual_tokens.device)).bool()
        # for masked tokens
        masked_indices = self.mask_token_id * torch.ones_like(visual_tokens, device=visual_tokens.device)
        # replace Yi with [mask] if m = 1, otherwise, when m = 0
        masked_tokens = mask * masked_indices + (~mask) * visual_tokens
        predicted_tokens = self.transformer(masked_tokens)  #transformer predict the probability of tokens
        
        logits = predicted_tokens
        z_indices = visual_tokens

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, mask_num, ratio):
        
        """ predict """
        masked_z_indices = mask * self.mask_token_id + (~mask) * z_indices
        logits = self.transformer(masked_z_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        prob = torch.softmax(logits, dim= -1)
        #Find max probability for each token value & max prob index
        z_indices_predict_prob, z_indices_predict = prob.max(dim=-1)
        
        """ mask schedule """
        # gamma(t / iter)
        mask_ratio = self.gamma(ratio)
        # number of taken to mask = ceil(mask_ratio * mask_num)
        mask_len = torch.ceil(mask_num * mask_ratio).long()

        """ sample """
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * g

        """ mask """
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        confidence[~mask] = torch.inf
        #sort the confidence for the rank 
        _, idx = confidence.topk(mask_len, dim=-1, largest=False)
        #define how much the iteration remain predicted tokens by mask scheduling
        mask_bc = torch.zeros(z_indices.shape, dtype=torch.bool, device= z_indices_predict_prob.device)
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        mask_bc = mask_bc.scatter_(dim= 1, index= idx, value= True)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
