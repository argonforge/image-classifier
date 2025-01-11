import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--arch', type=str, default='alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    if args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
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
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(2048, 500)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(500, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
        model.fc = fc
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()
    epochs = args.epochs
    print_every = 40
    steps = 0

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
    checkpoint = {'arch': args.arch,
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')

if __name__ == "__main__":
    main()
