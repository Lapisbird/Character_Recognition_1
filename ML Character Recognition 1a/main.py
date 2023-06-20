import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# This code is going to be VERY commented, as I am creating this as a process
# of learning, and thus I wish to be as detailed as possible in my understanding

# defining a batch size
batch_size = 64

# number of workers to use for parallelized data loading
num_workers = 0

# creates an instance of the torchvision.transforms.Compose() class, which
# allows one to create a transformation pipeline and do multiple transformations
# at once
transform = transforms.Compose([
    # converts the images to tensors
    transforms.ToTensor(),
    # normalizes the tensors for a mean of 0.5 and a std of 0.5
    transforms.Normalize(0.5, 0.5)
])

# Downloads the MNIST dataset and stores it in ./data
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,  # specifies the training portion of the MNIST dataset
    download=True,
    transform=transform  # transforms the image according to the transform defined earlier
)

# Creates a data loader
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,  # shuffle the dataset before each epoch
    num_workers=num_workers
)

testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

#create a new class inheriting from nn.Module
class CNN(nn.Module):
    def __init__(self): #constructor for this class
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #convolution layer with 1 input channel, 6 outputs, and 3x3 kernel. Results in tensor that is 26 x 26 x3
        self.relu = nn.ReLU() #non-linearity
        self.maxPool = nn.MaxPool2d(2, 2) #maxpool down to 13 x 13 x 3 with 2x2 kernel and stride of 2
        self.conv2 = nn.Conv2d(6, 18, 4) #second conv layer with 6 input channel, 18 output, 4x4 kernel. This will result in a 10 x 10 x 18 output
        #another ReLU will go here
        #we can use the same maxpool to reduce to a 5 x 5 x 18 tensor
        self.fc1 = nn.Linear(5*5*18, 128) #first fully connected layer which takes our flattened 5*5*18 tensor and outputs 128 elements
        #another relu goes here
        self.fc2 = nn.Linear(128, 64)
        #another relu here
        self.fc3 = nn.Linear(64, 10)

        #NOTE: For relu, F.relu can also be used directly instead of having to create an nn.ReLU object
        #the difference between F.relu and nn.ReLU is sort of like the differenct between a static method
        #and an instance method in Java, respectively

    def forward(self, x): #I could definitely significantly truncate the code below, but I think I'll keep it like this so that the process is very clear
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxPool(x)

        x = x.view(-1, 5*5*18) #flattening from 3d tensor to 1d

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

#creating our net
net = CNN()

#defining the loss function
criterion = nn.CrossEntropyLoss()
#We will use stochastic gradient descent
#net.parameters() returns all of the trainable parameters for our network
#a learning rate of 0.001 is implemented
#the momentum term is 0.9. Momentum can be essentially thought of as inertia during gradient descent. Helps to overcome local minima and oscillations
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

#number of epochs
num_epochs = 4

#use the GPU to train if possible. Else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + ("GPU" if torch.cuda.is_available() else "CPU"))
net.to(device)


for epoch in range(num_epochs):

    running_loss = 0.0

    #enumerate over each batch in our training data
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data[0].to(device), data[1].to(device) #make sure that our data is sent to GPU/CPU as appropriate
        optimizer.zero_grad() #reset the gradient so that the previous iteration does not affect the current one
        outputs = net(inputs) #run the batch through the curren model
        loss = criterion(outputs, labels) #calculate the loss
        loss.backward() #Using backpropagation, calculate the gradients
        optimizer.step() #Using the gradients, adjust the parameters (SGD with momentum in this case)
        running_loss += loss.item()

        #resets running loss every 2000 mini-batches
        #I'm a bit uncertain about this. To be technical, would this necessarily reset the running
        #loss every 2000 mini-batches? Because this assumes that there would then be exactly
        #an integer multiple of 2000 number of mini-batches, no? Otherwise, although we would
        #only every calculate the running loss percentage over 2000 mini-batches, they may not
        #necessarily be consecutive to each other
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


print("Training complete!")


#evalute model performance on test dataset overall

correct = 0
total = 0

with torch.no_grad(): #do not run gradient calculations
    for data in testloader: #for each data (image) in our test data set
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1) #We use "_" for the first value because we do
        #not care about it and wish to discard it. predicted then receives a tensor with the index
        #of the maximum value along dimension 1 in outputs.data. Basically the digit the model
        #believes the image is.
        #I do not fully understand the specifics of how torch.max(outputs.data, 1) works yet. Need to figure it out
        total += labels.size(0) #count the total number of elements here by finding the size of dim 0 of the labels tensor
        correct += (predicted == labels).sum().item() #we find the number of correct predictions by comparing
        #the predicted and labels tensor to create a boolean tensor that tells us if a certain
        #prediction was correct. We then sum it up to create a 0d tensor with a scalar value
        #(I presume we can do this because True == 1 and False == 0). Finally we use .item() to convert
        #the 0d tensor to a scalar value


accuracy = 100 * (correct/total)
print(f"Accuracy on the test set: {accuracy:.2f}%")


