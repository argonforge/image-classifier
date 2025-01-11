import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import json  

def process_image(image_path):
    image = Image.open(image_path)
    img = image.resize((256, 256))
    width = 256
    height = 256
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    img = np.array(img).transpose((2, 0, 1)) / 256
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = img - means
    img = img / stds
    img_tensor = torch.Tensor(img)
    return img_tensor

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(4096, 4096)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.5)),
                              ('fc3', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model.classifier = classifier
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = fc
    else:
        print(f"Architecture {arch} not recognized. Please use 'alexnet' or 'resnet50'.")
        return None

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()


    with open(args.category_names, 'r') as f:
        class_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    model = model.to(device)

    img = process_image(args.input)
    img = img.unsqueeze_(0)
    img = img.float()

    img = img.to(device)

    with torch.no_grad():
        output = model.forward(img)

    probability = F.softmax(output.data,dim=1)
    probs, classes = probability.topk(args.top_k)
    probs = [float(prob) for prob in probs[0]]
    classes = [int(_class) for _class in classes[0]]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[cls] for cls in classes]
    top_names = [class_to_name[cls] for cls in top_classes]  

    print('Probabilities:', probs)
    print('Classes:', top_classes)
    print('Names:', top_names)  

if __name__ == "__main__":
    main()
