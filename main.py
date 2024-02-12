import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18, DAResNet18, DGResNet18
# 1. Activation Shaping Module
from models.resnet import asm_hook
from models.resnet import register_forward_hooks, remove_forward_hooks

from globals import CONFIG

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    
    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)
    
    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model: BaseResNet18, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])

    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()

        print(f'EPOCH: {epoch+1}')

        # Register forward hooks
        if CONFIG.experiment in ['random']:
            hook_handles = []
            #hook_handles = register_forward_hooks(model, asm_hook, nn.ReLU) 
            hook_handles.append(model.resnet.layer1[0].relu.register_forward_hook(asm_hook))  
        elif CONFIG.experiment in ['domain_adaptation']:
            pass
            #hook_handles.append(model.resnet.layer1[0].bn1.register_forward_hook(model.rec_actmaps_hook))
            #hook_handles.append(model.resnet.layer1[0].relu.register_forward_hook(model.asm_source_hook))
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):

                if CONFIG.experiment in ['baseline', 'random']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                ######################################################
                #elif... TODO: Add here train logic for the other experiments
                ######################################################
                elif CONFIG.experiment in ['domain_adaptation']:
                    x_source, y_source, x_target = batch
                    x_source, y_source, x_target = x_source.to(CONFIG.device), y_source.to(CONFIG.device), \
                                                    x_target.to(CONFIG.device)
                    # register forward hooks, record activation maps and remove rec activation hook
                    hook_handles = []
                    #hook_handles += register_forward_hooks(model, model.rec_actmaps_hook, nn.ReLU, 2)
                    hook_handles.append(model.resnet.layer4[0].relu.register_forward_hook(model.rec_actmaps_hook))
                    model.record_activation_maps(x_target)
                    remove_forward_hooks(hook_handles)
                    # register forward hooks to multiply activation maps
                    #hook_handles += register_forward_hooks(model, model.asm_source_hook, nn.ReLU, 2)
                    hook_handles.append(model.resnet.layer4[0].relu.register_forward_hook(model.asm_source_hook))
                    loss = F.cross_entropy(model(x_source), y_source)
                    remove_forward_hooks(hook_handles)

                elif CONFIG.experiment in ['domain_generalization']:
                    (x1, y1), (x2, y2), (x3, y3) = batch
                    x1, y1 = x1.to(CONFIG.device), y1.to(CONFIG.device)
                    x2, y2 = x2.to(CONFIG.device), y2.to(CONFIG.device)
                    x3, y3 = x3.to(CONFIG.device), y3.to(CONFIG.device)
                    x = torch.cat([x1, x2, x3])
                    y = torch.cat([y1, y2, y3])

                    # Register forward hooks to record activation maps
                    hook_handles = []
                    #hook_handles.append(model.resnet.layer1[0].bn1.register_forward_hook(model.rec_actmaps_hook))
                    model.rec_actmaps(x1, x2, x3)
                    #remove_forward_hooks(hook_handles)
                    # Register forward hooks to forward pass
                    #hook_handles.append(model.resnet.layer1[0].bn1.register_forward_hook(model.asm_hook))
                    loss = F.cross_entropy(model(x), y)
                    #remove_forward_hooks(hook_handles)

            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()

            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()

        # Detach hooks
        if CONFIG.experiment in ['random', 'domain_adaptation']:
            remove_forward_hooks(hook_handles)
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch+1}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline', 'random']:
        model = BaseResNet18()

    ######################################################
    #elif... TODO: Add here model loading for the other experiments (eg. DA and optionally DG)

    ######################################################
        
    elif CONFIG.experiment in ['domain_adaptation']:
        model = DAResNet18()
    elif CONFIG.experiment in ['domain_generalization']:
        model = DGResNet18()
    
    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
