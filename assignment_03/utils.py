
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import random
import matplotlib.pyplot as plt
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.manifold import TSNE


def imshow(img, i=0, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
    """
    shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
    """
    # undoes the normalization
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def prevalidate(model, val_loader,criterion, device):
    ### START CODE HERE ### (≈ 12 lines of code)
    model.eval()
    correct, total = 0, 0
    loss_step = []
    with torch.no_grad():
        for data in val_loader:
            inp_data,labels = data
            inp_data, labels = inp_data.to(device), labels.to(device)
            outputs = model(inp_data)
            val_loss = criterion(outputs, labels)
            loss_step.append(val_loss.item())
        # dont forget to take the means here
        val_loss_epoch = np.mean(loss_step)
    ### END CODE HERE ###
        return val_loss_epoch

def pretrain_one_epoch(model, optimizer, train_loader, criterion, device):
    ### START CODE HERE ### (≈ 12 lines of code)
    model.train()
    loss_step = []
    for data in train_loader:
        # Move the data to the GPU
        inp_data, labels = data
        inp_data, labels = inp_data.to(device), labels.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_step.append(loss.item())
    # dont forget the means here
    loss_curr_epoch = np.mean(loss_step)
    ### END CODE HERE ###
    return loss_curr_epoch

def save_model(model, path, epoch, optimizer, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, path)

def pretrain(model, optimizer, num_epochs, train_loader, val_loader, criterion, device):
    dict_log = {"train_loss":[], "val_loss":[]}
    ### START CODE HERE ### (≈ 12 lines of code)
    best_val_loss = 1e8
    model = model.to(device)
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss_curr_epoch = pretrain_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss = prevalidate(model, val_loader, criterion, device)

        # Print epoch results to screen 
        msg = (f'Ep {epoch}/{num_epochs}: || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        pbar.set_description(msg)

        dict_log["train_loss"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)
        
        # Use this code to save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'best_model_min_val_loss.pth', epoch, optimizer, val_loss)   
        ### END CODE HERE ### 
    return dict_log


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss_step = []
    for data in val_loader:
        inp_data,labels = data
        inp_data = inp_data.to(device)
        labels = labels.to(device)
        outputs = model(inp_data)
        val_loss = criterion(outputs, labels)
        predicted = torch.max(outputs, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_step.append(val_loss.item())
    # dont forget to take the means here
    val_acc = (100 * correct / total).cpu().numpy() 
    val_loss_epoch = torch.tensor(loss_step).mean().numpy()
    return val_acc , val_loss_epoch

# Provided
@torch.no_grad()
def get_features(model, dataloader, device):
    model = model.to(device)
    model.eval()
    feats, labs = [], []
    for i in dataloader:
        inp_data,labels = i
        inp_data = inp_data.to(device)
        features = model(inp_data)
        features = features.cpu().detach().flatten(start_dim=1)
        labels = labels.cpu().detach()
        feats.append(features)
        labs.append(labels)
    f = torch.cat(feats, dim=0)
    l = torch.cat(labs, dim=0)
    return f,l


def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_step = []
    correct, total = 0, 0
    for data in train_loader:
        # Move the data to the GPU
        inp_data,labels = data
        inp_data = inp_data.to(device)
        labels = labels.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(loss.item())
    # dont forget the means here
    loss_curr_epoch = np.mean(loss_step)
    train_acc = (100 * correct / total).cpu()
    return loss_curr_epoch, train_acc


def linear_eval(model, optimizer, num_epochs, train_loader, val_loader, device):
    best_val_loss = float('inf')
    best_val_acc = 0
    model = model.to(device)
    dict_log = {"train_acc_epoch":[], "val_acc_epoch":[], "loss_epoch":[], "val_loss":[]}
    train_acc, _ = validate(model, train_loader, device)
    val_acc, _ = validate(model, val_loader, device)
    print(f'Init Accuracy of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss_curr_epoch, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_acc, val_loss = validate(model, val_loader, device)

        # Print epoch results to screen 
        msg = (f'Ep {epoch}/{num_epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        pbar.set_description(msg)
        # Track stats
        dict_log["train_acc_epoch"].append(train_acc)
        dict_log["val_acc_epoch"].append(val_acc)
        dict_log["loss_epoch"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_head_min_val_loss.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_head_max_val_acc.pth')
    return dict_log


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model


def default(val, def_val):
    return def_val if val is None else val

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


def tsne_plot_embeddings(features, labels, class_names, title="T-SNE plot"):
    plt.figure(figsize=(12, 12))
    latent_space_tsne = TSNE(2, verbose = True, n_iter = 2000, metric="cosine", perplexity=50, learning_rate=500)
    xa_tsne = latent_space_tsne.fit_transform(features.cpu().numpy()[:, :])
    colors = plt.rcParams["axes.prop_cycle"]()  
    for class_idx in range(len(class_names)):
        c = next(colors)["color"]
        plt.scatter(xa_tsne[:,0][labels==class_idx], xa_tsne[:,1][labels==class_idx], color=c, label=class_names[class_idx])

    plt.legend(class_names, fontsize=18, loc='center left', bbox_to_anchor=(1.05, 0.5))
    if title is not None:
        plt.title(title, fontsize=18)

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.savefig("tsne_plot_embeddings.png")
    plt.show()