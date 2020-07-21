
from settings import TOKENIZER, MAX_LEN
import torch



class prepare_dataset():
    def __init__(self, text, label):
        
        self.text = text
        self.label = label
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        
        text = self.text[idx]
        text = " ".join(text.split())
        label = self.label[idx]
        
        inputs = self.tokenizer.encode_plus(text , None,
                                           add_special_tokens=True,
                                           max_length = self.max_len,
                                           pad_to_max_length=True)
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        padding_length = self.max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'masks' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)
        }
        