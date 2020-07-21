import transformers




BERT_PATH = '../path to bert model'                          #'./bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
MAX_LEN = 192
BATCH_SIZE = 128
V_BATCH_SIZE = 32
TRAIN_PATH =  'path to train.csv'                     #'../train.csv'
EPOCHS = 8


# MODEL_PATH = ''     