import argparse
import os,sys,time
import numpy as np
import random
import pandas as pd
import datetime 
from tqdm import tqdm 
import torch 
import torchvision 
from torch import nn 
from torchvision import transforms, datasets
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_lightning import seed_everything
from copy import deepcopy
from torchmetrics import Accuracy


def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsize",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='experiments/classifier/',
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--es_monitor",
        type=str,
        default='val_acc',
    )
    parser.add_argument(
        "--es_mode",
        type=str,
        default='max',
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
    )
    opt = parser.parse_args()
    return opt

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{info} " + "=========="*6 + f" {nowtime}")
    # print(str(info)+"\n")

def recordlog(epoch, epochs, info, log_dir, show_time=True):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current = f"\nEpoch {epoch} / {epochs} " + "=========="*6 + f" {nowtime}"
    with open(log_dir, 'a') as f:
        if show_time: f.write(current + '\n')
        f.write(str(info) + '\n')

def fix_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Fix seed as {seed}, and set deterministic as True.')
    
class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
      
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main(opt):
    fix_seed(42)
    bsize = opt.bsize
    epochs = opt.epochs
    nw = opt.num_workers
    lr = opt.lr
    weight_decay = opt.weight_decay
    num_classes = 4
    
    print('=== 0 Prepare dirs ===')
    # make dirs, create opt.model_dir if not exist, and create subdirs for multiple runs as 1, 2, 3, ...
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M')
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    root_dir = opt.model_dir + f"{str(len(os.listdir(opt.model_dir)))}_{current_time}"
    model_dir = root_dir
    log_dir = os.path.join(root_dir, 'logs.txt')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    print('=== 1 Logging info ===')
    # log command line and time and opts
    with open(log_dir, 'a') as f:
        f.write(f'{current_time}\n')
        f.write('CMD: ' + ' '.join(sys.argv) + '\n')
        f.write(f'Arguments: {opt}\n')
    
    
    print('=== 2 Prepare dataset and dataloader ===')
    transform = transforms.Compose([transforms.ToTensor()])

    ds = datasets.ImageFolder(root='datasets/halftones', transform=transform, target_transform=None)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [int(len(ds)*0.8), int(len(ds)*0.1), int(len(ds)*0.1)])
    dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=bsize, shuffle=True, num_workers=nw, pin_memory=True)
    dl_val =  torch.utils.data.DataLoader(ds_val, batch_size=bsize, shuffle=False, num_workers=nw, pin_memory=True)
    dl_test =  torch.utils.data.DataLoader(ds_test, batch_size=bsize, shuffle=False, num_workers=nw, pin_memory=True)

    print(f'Length of train, valid and test: {len(ds_train)}, {len(ds_val)}, {len(ds_test)}')
    features, labels = next(iter(dl_train))
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    
    
    print('=== 3 Initialise model, loss_fn and optimiser ===')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ResNet(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    metrics_dict = {"acc":Accuracy(task='multiclass', num_classes=num_classes)}
    
    #early_stopping相关设置
    monitor=opt.es_monitor
    patience=opt.es_patience
    mode=opt.es_mode
    
    print('=== 3 Train model ===')
    history = {}
    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))
        
        
        # Train
        model.train()
        total_loss, step = 0, 0
        iterator = tqdm(dl_train, desc=f"Train epoch {epoch}: ", disable=False)
        train_metrics_dict = deepcopy(metrics_dict) 
        
        for i, batch in enumerate(iterator):
            data, target = batch
            data, target = data.to(device), target.to(device)
            
            optimiser.zero_grad()
            preds = model(data.to(device))
            loss = loss_fn(preds, target)
            loss.backward()
            optimiser.step()
            
            step_metrics = {"train_"+name:metric_fn(preds.cpu(), target.cpu()).item() 
                        for name,metric_fn in train_metrics_dict.items()}
            
            step_log = dict({"train_loss":loss.item()},**step_metrics)
            
            total_loss += loss.item()
            step += 1
            if i!=len(dl_train)-1:
                iterator.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {"train_"+name:metric_fn.compute().item() 
                                for name,metric_fn in train_metrics_dict.items()}
                epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
                iterator.set_postfix(**epoch_log)
                recordlog(epoch, epochs, epoch_log, log_dir)

                for name,metric_fn in train_metrics_dict.items():
                    metric_fn.reset()
        
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]
        
        
        # Validation
        model.eval()
        total_loss, step = 0, 0
        iterator = tqdm(dl_val, desc=f"Valid epoch {epoch}: ", disable=False)
        val_metrics_dict = deepcopy(metrics_dict) 
        
        with torch.no_grad():
            for i, batch in enumerate(iterator): 
                data,target = batch
                data, target = data.to(device), target.to(device)
                
                #forward
                preds = model(data.to(device))
                loss = loss_fn(preds,target)

                #metrics
                step_metrics = {"val_"+name:metric_fn(preds.cpu(), target.cpu()).item() 
                                for name,metric_fn in val_metrics_dict.items()}

                step_log = dict({"val_loss":loss.item()},**step_metrics)

                total_loss += loss.item()
                step+=1
                if i!=len(dl_val)-1:
                    iterator.set_postfix(**step_log)
                else:
                    epoch_loss = (total_loss/step)
                    epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                    for name,metric_fn in val_metrics_dict.items()}
                    epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                    iterator.set_postfix(**epoch_log)
                    recordlog(epoch, epochs, epoch_log, log_dir, show_time=False)

                    for name,metric_fn in val_metrics_dict.items():
                        metric_fn.reset()
                        
        epoch_log["epoch"] = epoch           
        for name, metric in epoch_log.items():
            history[name] = history.get(name, []) + [metric]
            
            
        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:
            torch.save(model.state_dict(),os.path.join(model_dir, "best_model.pt"))
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                arr_scores[best_score_idx]),file=sys.stderr)
        if len(arr_scores)-best_score_idx>patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor,patience),file=sys.stderr)
            model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
            break 
        

def test(opt, model_dir):
    fix_seed(42)
    opt = parsr_args()
    bsize = opt.bsize
    nw = opt.num_workers
    
    ds = datasets.ImageFolder(root='datasets/halftones', transform=transforms.ToTensor(), target_transform=None)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [int(len(ds)*0.8), int(len(ds)*0.1), int(len(ds)*0.1)])
    dl_test =  torch.utils.data.DataLoader(ds_test, batch_size=bsize, shuffle=False, num_workers=nw)
    features, labels = next(iter(dl_test))
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ResNet(num_classes=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    state_dict = torch.load(model_dir)
    print(model.load_state_dict(state_dict, strict=False))
    
    metrics_dict = {"acc":Accuracy(task='multiclass', num_classes=4)}
    
    # Testing
    model.eval()
    total_loss, step = 0, 0
    iterator = tqdm(dl_test, desc=f"Testing: ", disable=False)
    val_metrics_dict = deepcopy(metrics_dict) 
    
    # initialise prediction and target tensors
    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    
    with torch.no_grad():
        for i, batch in enumerate(iterator): 
            data,target = batch
            data, target = data.to(device), target.to(device)
            
            #forward
            preds = model(data.to(device))
            loss = loss_fn(preds,target)
            
            # append predictions and targets to tensors
            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            all_targets = torch.cat((all_targets, target.cpu()), dim=0)

            #metrics
            step_metrics = {"val_"+name:metric_fn(preds.cpu(), target.cpu()).item() 
                            for name,metric_fn in val_metrics_dict.items()}

            step_log = dict({"val_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            step+=1
            if i!=len(dl_test)-1:
                iterator.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss/step)
                epoch_metrics = {"val_"+name:metric_fn.compute().item() 
                                for name,metric_fn in val_metrics_dict.items()}
                epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                iterator.set_postfix(**epoch_log)
    # get the confusion matrix
    preds = all_preds.argmax(dim=1)
    targets = all_targets
    print(f'Predictions shape: {preds.shape}, Targets shape: {targets.shape}')
    # draw confusion matrix using sklearn
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3'])
    disp.plot()
    plt.savefig('confusion_matrix.png')


if __name__ == '__main__':
    opt = parsr_args()
    if opt.mode == 'train':
        main(opt)
    elif opt.mode == 'test':
        test(opt, 'experiments/classifier/0_2023_07_16_18:56/best_model.pt')
    else:
        pass




    
