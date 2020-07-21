import pandas as pd
import numpy as np 
from sklearn import metrics, model_selection
import os

import torch
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup


import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings

warnings.filterwarnings("ignore")

from settings import TRAIN_PATH
from settings import BATCH_SIZE, V_BATCH_SIZE

from model import BertBaseUncased
from data_gen import prepare_dataset
from engine import train_loop, eval_loop




def ohe(df,target_col):
    
    encoded = pd.get_dummies(df.sort_values(by=[target_col])[target_col])
    
    df = df.join(encoded)
    
    return df
    


def train():
    
    df = pd.read_csv(TRAIN_PATH).fillna('None')
    
    train, valid = model_selection.train_test_split(df, test_size = 0.15, random_state=42, stratify=df['Class Index'].values)
    
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    
    train= ohe(train, 'Class Index')
    valid = ohe(valid, 'Class Index')
    
    train_labels = train[train.columns[-4:]].values
    valid_labels = valid[valid.columns[-4:]].values
    
    
    train_data = prepare_dataset(text=train['Description'].values,
                                label=train_labels)
    
    valid_data = prepare_dataset(text=valid['Description'].values,
                                label=valid_labels)
    
    
    train_sampler = torch.utils.data.DistributedSampler(train_data,
                                                       num_replicas=xm.xrt_world_size(),
                                                       rank= xm.get_ordinal(),
                                                       shuffle=True)

    valid_sampler = torch.utils.data.DistributedSampler(valid_data,
                                                       num_replicas=xm.xrt_world_size(),
                                                       rank= xm.get_ordinal(),
                                                       shuffle=False)
    
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,num_workers=4,sampler=train_sampler,drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data,batch_size=V_BATCH_SIZE,num_workers=4,sampler=valid_sampler,drop_last=True)
    
    
    
#     device= torch.device('cuda')
    
    device = xm.xla_device()
        

    model = BertBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    num_train_steps = int(len(train_data)/BATCH_SIZE/xm.xrt_world_size() * EPOCHS)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')
    
    lr = 3e-4 * xm.xrt_world_size()
    
    optimizer = AdamW(optimizer_parameters,lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    best_acc=0
    
    for epoch in range(EPOCHS):
        
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        
        train_loss = train_loop(para_loader.per_device_loader(device),model=model, optimizer=optimizer,scheduler=scheduler,device=device)
        
        para_loader = pl.ParallelLoader(valid_dataloader, [device])
        
        val_acc, val_loss = eval_loop(para_loader.per_device_loader(device), model, device)
        
#         print(f"EPOCH: {epoch} train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")
        
        if val_acc > best_acc:
            torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},'best_model.bin')
            
            best_acc=val_acc
            
        
        xm.master_print(f'Epoch: {epoch+1} train_loss: {train_loss} val_loss: {val_loss} Accracy: {val_acc}')
            
            
        
        
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = train()

Flags ={}
xmp.spawn(_mp_fn,args=(Flags,), nprocs=1,start_method='fork')        
