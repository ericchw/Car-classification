# Accuracy: ~95% +-1%
# Run script: python train.py --model_path ./model.pth --train_data_path ./DATA/train --val_data_path ./DATA/val  --csv_path ./DATA/number_of_samples.csv --log_txt_path ./train.txt --result_png_path ./train_model.png

# Ref: https://www.youtube.com/watch?v=5rD8f1oiuWM
# Ref:https://ithelp.ithome.com.tw/articles/10218698 
# ES: https://www.kaggle.com/general/178486
# Torch ResNet18 sample for santa: https://www.youtube.com/watch?v=5rD8f1oiuWM

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
# import seaborn as sns

# Define the command line arguments
parser = argparse.ArgumentParser(description='Testing a pre-trained PyTorch model on a folder of images')
parser.add_argument('--model_path', type=str, default='./model.pth', help='path save the pre-trained model')
parser.add_argument('--train_data_path', type=str, default='./DATA/train', help='path to the train data folder with images')
parser.add_argument('--val_data_path', type=str, default='./DATA/val', help='path to the validation data folder with images')
parser.add_argument('--csv_path', type=str, default='./DATA/number_of_samples.csv', help='path to csv file with number of files for each class into train, validation, and test sets')
parser.add_argument('--log_txt_path', type=str, default='./train.txt', help='path to save epochs result')
parser.add_argument('--result_png_path', type=str, default='./train.png', help='path to save train model plt result')
args = parser.parse_args()

# # Set the device to be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'GPU/CPU: {torch.cuda.is_available()}')
# print(f'Device: {device}')

# Define the hyperparameters for training
batch_size = 128 #64 if overload
num_epochs = 50
learning_rate = 0.0001
es_target = 5

# Define the transforms to be applied to the data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor(),  # convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image which is commonly used value for ResNet model
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 224x224
    transforms.ToTensor(),  # convert the image to a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image which is commonly used value for ResNet model
])

# Load the data
train_data = datasets.ImageFolder(args.train_data_path, transform=train_transforms)
val_data = datasets.ImageFolder(args.val_data_path, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# Visualising and preparing data and Load the data into a pandas DataFrame
df = pd.read_csv(args.csv_path)

# Calculate the number of samples in the train and validation set
num_train_samples = df['train'].sum()
num_val_samples = df['val'].sum()
num_classes = len(np.unique(df['Name of class']))
# print(f'num_classes: {num_classes}')

# sns.set_style('darkgrid')
# plt.figure(figsize=(12,7))
# sns.barplot(x=df['Number of files'],y=df['Name of class'])


# Define the model
model = models.resnet152(pretrained=True)
num_input_features = model.fc.in_features
print(f'num_input_features: {num_input_features}')
model.fc = nn.Sequential(
    nn.Linear(num_input_features, 512), #in_fc, out_fc
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
model = model.to(device)

summary(model, (3, 224, 224)) # input shape of (RGB, H, W)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001) # large weights, which can help prevent overfitting
criterion = nn.CrossEntropyLoss()

# Train the model
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0.0
epochs = 0
es = 5 #EarlyStopping to prevent overfitting

open(args.log_txt_path, 'w') #cleanup txt
for epoch in range(num_epochs):
    epochs += 1
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0

    # Train the model on the training set
    model.train()
    for images, labels in train_loader: # move the data to GPU
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == labels).sum().item()

    train_loss /= num_train_samples
    train_acc = train_correct / num_train_samples

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs.data, 1)
            val_correct += (predictions == labels).sum().item()

    val_loss /= num_val_samples
    val_acc = val_correct / num_val_samples

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Print and log into txt with the progress result for each epoch
    print('Epoch [{}/{}], Batch Size [{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
          epochs, num_epochs, batch_size, train_loss, train_acc, val_loss, val_acc))
    open(args.log_txt_path, 'a').write('Epoch [{}/{}], Batch Size [{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}\n'.format(
          epochs, num_epochs, batch_size, train_loss, val_loss, train_acc, val_acc))
    
# Save the model if the validation accuracy is improved with Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        es = 0
        # torch.save(model.state_dict(), args.model_path)        
    else:
        es += 1
        print("Early stopping counter {} of 5".format(es))
    if es >= es_target:
        print("Early stopping with best_val_acc: ", best_val_acc, "and val_acc for this epoch: ", val_acc, "...")
        break
print("Model saved with validation accuracy: {:.4f}".format(round(best_val_acc * 100, 2)))

# Plot train val losses
fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(10, 5))
epochList = range(1, epochs + 1) #Display epochs start from 1, but not 0 in plt

plt1.plot(epochList, train_losses, label='Train Loss')
plt1.plot(epochList, val_losses, label='Val Loss')
plt1.set_title('Train/Val Loss')
plt1.set_xlabel('Epochs')
plt1.set_ylabel('Loss')
plt1.set_ylim([0, 3.0])
plt1.legend(loc='best')

plt2.plot(epochList, train_accs, label='Train Acc')
plt2.plot(epochList, val_accs, label='Val Acc')
plt2.set_title('Train/Val Accuracy')
plt2.set_xlabel('Epochs')
plt2.set_ylabel('Accuracy')
plt2.legend(loc='best')

fig.suptitle("Model validation accuracy: {:.2f}%".format(round(best_val_acc * 100, 2)), fontsize=14, fontweight='bold')

plt.savefig(args.result_png_path)
plt.show()