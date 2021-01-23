
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
from PIL import Image

def label_mapping(filename):
    with open(os.getcwd() + '/' + filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epoch']
    model = checkpoint['model']
    model.classifier = checkpoint ['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    # Process a PIL image for use in a PyTorch model
    pil_image_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    return pil_image_transform(pil_image)

def predict(device, image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # code to predict the class from an image file

    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze_(0)
    
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        log_predictions = model.forward(image)
    ps = torch.exp(log_predictions)
    top_p, top_k = ps.topk(topk, dim=1)

    top_p = top_p.cpu()
    top_k = top_k.cpu()
    
    top_p = top_p.numpy()
    top_k = top_k.numpy()
    
    top_p = top_p.tolist()[0]
    top_k = top_k.tolist()[0]
    
    map_idx_class = {val: key for key, val in model.class_to_idx.items()}
    top_c = [map_idx_class [item] for item in top_k]
    
    return np.array(top_p), np.array(top_c)

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', dest='is_gpu', default=False)
options = parser.parse_args()

device = 'cuda' if options.is_gpu else 'cpu'

category_to_name = label_mapping(options.category_names)
model = load_checkpoint(options.checkpoint)
probs, classes = predict(device, options.image_path, model, options.top_k)
flower_names = [category_to_name[item] for item in classes]

print("Image {} of a '{}' is predicted as follows".format(options.image_path, category_to_name[options.image_path.split('/')[-2]]))
print("Most probably: {}".format(flower_names[np.argmax(probs)]))
if category_to_name[options.image_path.split('/')[-2]] == flower_names[np.argmax(probs)]:
    print("Prediction is correct")
else:
    print("Prediction is incorrect")
print("Probability  :  Flower")
for i in range(len(flower_names)):
    print("{:3.3f} : {}".format(probs[i] * 100, flower_names[i]))
