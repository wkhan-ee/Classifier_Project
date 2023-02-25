import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os.path import isdir
import argparse


def arg_parser():
    # create an ArgumentParser object with a description of the program
    parser = argparse.ArgumentParser(description='Setting for NN training')
    
    # add arguments to the parser
    parser.add_argument('data_dir', type=str, default='flowers', help='Directory of the data')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU')
    
    # parse the arguments and return them as a Namespace object
    return parser.parse_args()
# Function for data transformation
def data_transform(train_dir,valid_dir,test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return train_data, valid_data, test_data


def data_loader(data, train=True, batch_size=50, shuffle=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA was not found on device, using CPU instead.")
    return device

def load_pretrained_model(arch):
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        print("Model not supported. Please choose from: vgg16 or densenet121")
    
    model.name = arch
    #Freeze the non classifier parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def classifier(model, hidden_units):

    input_units = model.classifier[0].in_features #input units of the classifier
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_units, hidden_units, bias=True)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    return classifier

def train(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):

    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(outputs)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                    
                print('Epoch: {}/{}...'.format(e+1, epochs),
                    'Training Loss: {:.3f}...'.format(running_loss/print_every),
                    'Validation Loss: {:.3f}...'.format(valid_loss/len(validloader)),
                    'Validation Accuracy: {:.3f}'.format(accuracy/len(validloader)))
                running_loss = 0
                model.train()

    return model        

def test_model(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

def save_checkpoint(model, train_data, save_dir):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': model.name,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)


def main():

    args = arg_parser()
    
    #assigment of arguments
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    #transforming the data and loading it
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data, valid_data, test_data = data_transform(train_dir,valid_dir,test_dir)
    
    train_dataloader = data_loader(train_data)
    valid_dataloader = data_loader(valid_data, train=False)
    test_dataloader = data_loader(test_data, train=False)

    device = check_gpu(gpu)

    model = load_pretrained_model(arch)
    model.classifier = classifier(model, hidden_units)
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #training the model
    print_every = 30
    steps = 0
    updated_model = train(model, train_dataloader, valid_dataloader, device, criterion, optimizer, epochs, print_every, steps)

    #testing the model
    test_model(updated_model, test_dataloader, device)

    #saving the model
    save_checkpoint(updated_model, train_data, save_dir)


if __name__ == '__main__':
    main()
