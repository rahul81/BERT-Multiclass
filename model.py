
import transformers
from settings import BERT_PATH


class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(768,4)
        
    def forward(self,ids, masks, token_type_ids):
        _, out = self.bert(ids, masks, token_type_ids)
        out = self.dropout(out)
        out = self.out(out)
        
        return out      