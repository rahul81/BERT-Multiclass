import torch
import torch.nn as nn




def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)


def train_loop(dataloader, model ,optimizer, device,scheduler):
    
    model.train()
    
    epoch_loss = 0 
    
    counter = 0 
    for idx, batch in enumerate(dataloader):
        counter +=1
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids =  batch['token_type_ids']
        targets = batch['targets']
        
        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids= token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype= torch.float)
        
        
        optimizer.zero_grad()
        
        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)
        
        loss = loss_fn(outputs,targets)
        loss.backward()   
        
        xm.optimizer_step(optimizer)
#         optimizer.step()

        scheduler.step()
    
    
        if idx %50 == 0:
            xm.master_print(f'Batch: {idx} train_loss: {loss.item()}')
        
        epoch_loss+=loss.item()
        

        
    return epoch_loss/counter
        
        
        
        
        
def eval_loop(dataloader, model , device):
    
    model.eval()  
    epoch_acc = 0
    epoch_loss= 0
    counter = 0 
    for idx, batch in enumerate(dataloader):
        counter +=1
        ids = batch['ids']
        masks = batch['masks']
        token_type_ids =  batch['token_type_ids']
        targets = batch['targets']
        
        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        token_type_ids= token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype= torch.float)
        
        
        
        outputs = model(ids=ids, masks=masks, token_type_ids=token_type_ids)
        
        loss = loss_fn(outputs,targets)
        
        outputs = torch.argmax(outputs,axis=1)
        targets = torch.argmax(targets,axis=1)

        
        acc = metrics.accuracy_score(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy())
        
        epoch_acc+=acc
        epoch_loss+= loss.item()
    
    final_acc = epoch_acc/counter
    epoch_loss = epoch_loss/counter
    return final_acc, epoch_loss
        
        
        
