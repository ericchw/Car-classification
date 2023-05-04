# Run script: python test_model.py --model_path ./model_Ver3_ES5_test_Acc_9485.pth --test_data_path ./DATA/test --output_txt_path ./output.txt  --result_png_path ./test_model.png
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
# import sys

# Define the command line arguments
parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--model_path', type=str, default='./model.pth', help='path load the pre-trained model')
parser.add_argument('--test_data_path', type=str, default='./DATA/test', help='path to the test data folder with images')
parser.add_argument('--output_txt_path', type=str, default='./output.txt', help='path to save output txt')
parser.add_argument('--result_png_path', type=str, default='./test_model.png', help='path to save test model plt result')
args = parser.parse_args()

# Set the device to be used for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'GPU/CPU: {torch.cuda.is_available()}')
# print(f'Device: {device}')

# Load the test filenames
test_folder = args.test_data_path
test_filenames = []
for root, dirs, files in os.walk(test_folder):
    for file in files:
        # if file.endswith(".jpg"):
            test_filenames.append(os.path.join(root, file))

# Get the names of the subdirectories in the test folder
class_names = os.listdir(test_folder)
class_names.sort()
# Create a dictionary to map class indices to class names
class_mapping = {i: class_names[i] for i in range(len(class_names))}
num_classes = len(class_names)
print(class_mapping)
print('Total number of class: {len(class_names)}')

# Define the transforms to be applied to the data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor(),  # convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the image
])

# Load the model
model = models.resnet152(pretrained=True)
num_input_features= model.fc.in_features
model.fc = nn.Sequential( #need same as training
    nn.Linear(num_input_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes),
)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
model.to(device)

# Classify the test images and write the predictions to output.txt
class_correct = list(0. for i in range(len(class_names)))
class_total = list(0. for i in range(len(class_names)))
output_txt = open(args.output_txt_path, 'w')
with torch.no_grad(), output_txt:
    for filename in test_filenames:
        image = Image.open(filename)
        image = test_transforms(image)
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, prediction = torch.max(output, 1)
        predicted_class_name = class_mapping[prediction.item()]
        output_txt.write('{}: {}\n'.format(filename, predicted_class_name))
        
        # Update class accuracy
        true_class_name = os.path.basename(os.path.dirname(filename))
        true_class_idx = class_names.index(true_class_name)
        class_correct[true_class_idx] += int(predicted_class_name == true_class_name)
        class_total[true_class_idx] += 1

# Plot the results
fig = plt.figure(figsize=(20, 20))
overall_correct = sum(class_correct)
overall_total = sum(class_total)
overall_accuracy = overall_correct / overall_total
plt.suptitle('Overall Accuracy: {:.2f}%'.format(round(overall_accuracy * 100, 2)), fontsize=30)
for i in range(len(class_names)):
    dir_path = os.path.join(test_folder, class_names[i])
    image_files = os.listdir(dir_path)
    random_file = random.choice(image_files)
    image_path = os.path.join(dir_path, random_file)
    image = Image.open(image_path)
    plt.subplot(5, 4, i+1)
    plt.imshow(image)
    plt.title('{}: {:.2f}%'.format(class_names[i], round((class_correct[i] / class_total[i]) * 100, 2)))
    plt.axis('off')

plt.savefig(args.result_png_path)
plt.show()