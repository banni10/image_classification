import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from models import *

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(784,64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(64,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)
    
def classify_mlp_image(image):
    # Load the pretrained model
    # args=argparse.Namespace(epochs=30, num_users=100, frac=0.1, local_ep=10, local_bs=10, lr=0.01, momentum=0.5, model='cnn', kernel_num=9, kernel_sizes='3,4,5', num_channels=1, norm='batch_norm', num_filters=32, max_pool='True', dataset='mnist', num_classes=10, gpu=None, optimizer='sgd', iid=1, unequal=0, stopping_rounds=10, verbose=1, seed=1)

    model = MLP() # we do not specify ``weights``, i.e. create untrained model
    # print(model)
    model.load_state_dict(torch.load('model_mlp.pth'))
    model.eval()

    # Define image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
    ])

    # Load and preprocess the grayscale image
    # image_path = '/content/drive/MyDrive/Gargi/img6.png'
    # image = Image.open(image_path).convert('L')  # Open image in grayscale mode
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Print the image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Forward pass through the pretrained model
    with torch.no_grad():
        output = model(input_tensor)

    # Process the model output
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class

    # print('Predicted class:', predicted_class)
    # print('Class probabilities:', probabilities)
    

def classify_cnn_image(image):
    # Load the pretrained model
    # args=argparse.Namespace(epochs=30, num_users=100, frac=0.1, local_ep=10, local_bs=10, lr=0.01, momentum=0.5, model='cnn', kernel_num=9, kernel_sizes='3,4,5', num_channels=1, norm='batch_norm', num_filters=32, max_pool='True', dataset='mnist', num_classes=10, gpu=None, optimizer='sgd', iid=1, unequal=0, stopping_rounds=10, verbose=1, seed=1)

    model = CNNMnist() # we do not specify ``weights``, i.e. create untrained model
    # print(model)
    model.load_state_dict(torch.load('model_cnn.pth'))
    model.eval()

    # Define image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
    ])

    # Load and preprocess the grayscale image
    # image_path = '/content/drive/MyDrive/Gargi/img6.png'
    # image = Image.open(image_path).convert('L')  # Open image in grayscale mode
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Print the image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Forward pass through the pretrained model
    with torch.no_grad():
        output = model(input_tensor)

    # Process the model output
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class

    # print('Predicted class:', predicted_class)
    # print('Class probabilities:', probabilities)
    