import sys
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
import json
import os

def banner_msg(options):
    device = 'cuda' if options.is_gpu else 'cpu'
    print('Training model {!r} with {!r} hidden units for {!r} epochs on {}'.format(options.model_arch,
                                                                                    options.hidden_units,
                                                                                    options.epochs,
                                                                                    device))
    print('Learning rate is set to {!r}'.format(options.lr))
    print('Saving checkpoint in directory {!r}'.format(options.save_dir))
    print('Using directories {} & {} for training and validation images respectively'.format(options.dirname + '/train',
                                                                                             options.dirname + '/valid'))
    return

def get_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    # transforms for the training set
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Load the training datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True)

    # Validation
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    validation_loaders = torch.utils.data.DataLoader(validation_datasets, batch_size=64, shuffle = True)
    return train_dataloaders, validation_loaders, train_datasets

def train(options, train_dataloaders, validation_loaders):
    device = 'cuda' if options.is_gpu else 'cpu'
    
    model = None

    if options.model_arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif options.model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif options.model_arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    else:
        print("Defaulting to model arch: {}".format(options.model_arch))
        model = models.vgg19_bn(pretrained=True)
    
    print(model.classifier)
    
    in_features = model.classifier[0].in_features
    hidden_units = options.hidden_units
    out_classes = 102
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout (p = 0.5)),
                                            ('fc2', nn.Linear(hidden_units, hidden_units)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout (p = 0.5)),
                                            ('fc3', nn.Linear(hidden_units, out_classes)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier

    print(model.classifier)
    
    criterion = nn.NLLLoss()
    print(model.classifier.parameters())
    optimizer = optim.Adam(model.classifier.parameters(), lr=options.lr)


    model.to(device)
    epochs = options.epochs

    for e in range(epochs):
        running_train_loss = 0
    
        for images, labels in train_dataloaders:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            predictions = model.forward(images)
            train_loss = criterion(predictions, labels)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        
        else:
            validation_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for images, labels in validation_loaders:
                    images, labels = images.to(device), labels.to(device)

                    log_predictions = model.forward(images)
                    loss = criterion(log_predictions, labels)

                    validation_loss += loss.item()

                    ps = torch.exp(log_predictions)
                    top_p, top_k = ps.topk(1, dim=1)
                    equals = top_k == labels.view(*top_k.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_train_loss/len(train_dataloaders)),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loaders)))
                
            model.train()

    return model, optimizer

def save_checkpoint(model, save_dir, epochs, optimizer):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    except:
        raise
        
    # Save to the checkpoint
    checkpoint = {
        'epoch': epochs,
        'output_sz': 102,
        'model': model,
        'classifier': model.classifier,
        'model_state': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    
    return

parser = argparse.ArgumentParser()
parser.add_argument('dirname')
parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='save_dir')
parser.add_argument('--arch', action='store', dest='model_arch', type=str, default='vgg19_bn')
parser.add_argument('--learning_rate', action='store', dest='lr', type=float, default=0.001)
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=4096)
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=10)
parser.add_argument('--gpu', action='store_true', dest='is_gpu', default=False)
options = parser.parse_args()

banner_msg(options)
train_dataloaders, validation_loaders, train_datasets = get_dataloaders(options.dirname)
model, optimizer = train(options, train_dataloaders, validation_loaders)

model.class_to_idx = train_datasets.class_to_idx
save_checkpoint(model, options.save_dir, options.epochs, optimizer)