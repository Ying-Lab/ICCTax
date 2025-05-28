import torch
import torch.nn as nn
from model.model_hyena import HyenaDNAPreTrainedModel
from hyena.standalone_hyenadna import SequenceDecoder
import sys
import warnings

sys.path.append("/workspace/embedding")
pretrained_model_name = 'hyenadna-small-32k-seqlen'
warnings.filterwarnings("ignore", category=FutureWarning)

class HyenaModel(nn.Module):
    def __init__(self,lists):
        super(HyenaModel, self).__init__()
        self.model = HyenaDNAPreTrainedModel.from_pretrained(
           '/home/gyc/ICCTax/hyena/',
           pretrained_model_name,
        )
        self.supk_decoder = SequenceDecoder(d_model=256, d_output=lists[0], l_output=0, mode="pool")
        self.phyl_decoder = SequenceDecoder(d_model=256+lists[0], d_output=lists[1], l_output=0, mode="pool")
        self.genus_decoder = SequenceDecoder(d_model=256+lists[1], d_output=lists[2], l_output=0, mode="pool")
        
        self.dna_adaptation1 = nn.Linear(256,768)
        self.dna_adaptation2 = nn.Linear(768, 512)
        self.dna_adaptation3 = nn.Linear(512, 256)
        self.dna_adaptation4 = nn.Linear(256, 256)
        
        self.Supk1= nn.Linear(512,512)
        self.Supk2 = nn.Linear(256,lists[0])
        
        self.Phyl1 = nn.Linear(512,512)
        self.Phyl2 = nn.Linear(256,lists[1])
        self.Phyl3 = nn.Linear(256+lists[0],lists[1])
        
        self.Genus1 = nn.Linear(512,512)
        self.Genus2 = nn.Linear(256,lists[2])
        self.Genus3 = nn.Linear(256+lists[1],lists[2])
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, dna):
        dna_embed = self.get_dna(dna)
        
        supk_output=self.supk_decoder(dna_embed)
        supk_soft=nn.Softmax(dim=1)(supk_output)
        supk_emd=supk_soft.unsqueeze(1).repeat(1, 1502, 1)
        
        phyl_input=torch.cat((dna_embed, supk_emd), dim=-1)
        phyl_output=self.phyl_decoder(phyl_input)
        phyl_soft=nn.Softmax(dim=1)(phyl_output)
        phyl_emd=phyl_soft.unsqueeze(1).repeat(1, 1502, 1)
        
        genus_input=torch.cat((dna_embed, phyl_emd), dim=-1)
        genus_output=self.genus_decoder(genus_input)
        
        return dna_embed,supk_output,phyl_output,genus_output
    
    def get_dna(self,dna_inputs):
        dna_embed = self.model(dna_inputs)
        return dna_embed
    

