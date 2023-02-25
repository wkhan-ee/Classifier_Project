import torch
from torchvision import models
import argparse
import json
from PIL import Image
import numpy as np
from math import ceil

def arg_parser():
    # create an ArgumentParser object with a description of the program
    parser = argparse.ArgumentParser(description='Setting for model prediction')
    
    # add arguments to the parser
    parser.add_argument('--image', type=str, help='Image to predict', required=True)
    parser.add_argument('--checkpoint', type=str, help='Directory for the saved model', required=True)
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File that maps the class values to     other category names')
    parser.add_argument('--gpu', action="store_true", help='Use GPU')

    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load('checkpoint.pth')
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    with Image.open(image) as img:
        img = img.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img = np.array(img)/255
        img = (img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        img = img.transpose((2,0,1))

        return img

def predict(image, model, device, cat_to_name, top_k):
    model.eval()
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0)
    model.cpu()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[x] for x in top_class]
        top_flowers = [cat_to_name[x] for x in top_class]
        
        return top_p, top_class, top_flowers


def main():

    args = arg_parser()
    image = args.image
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(checkpoint)

    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    image = process_image(image)

    probs, classes, flowers = predict(image, model, device, cat_to_name, top_k)

    for i, j in zip(probs, flowers):
        print('Flower: {} Probability: {}'.format(j, i))
 
if __name__ == '__main__':
    main()
