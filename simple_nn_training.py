from torch.autograd.grad_mode import F
import torch.nn.functional as f
import torch
from torch import nn, optim, t
import torch.nn.functional as F
from torch.utils import data
import torchvision
import numpy as np
import sys
from torch.utils.data import SubsetRandomSampler

EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_TO_USE = 'B'
IMAGE_SIZE = 28*28
NUMBER_OF_LABELS =10

#at first we need to load the data:
test_x = sys.argv[3]
test_x = np.loadtxt(test_x) / 255
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_set = torchvision.datasets.FashionMNIST(root='./data',train=True, download=True,transform=transforms)
test_x_loader = transforms(test_x)[0].float()
# Splits train data set to the corresponding validation ratio.
train_set_size = len(train_set)
indices = list(range(train_set_size))
split = int(0.2 * train_set_size)
valid_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(valid_idx))
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_set,batch_size=64,sampler=validation_sampler)

"""
   test_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch,
                                             shuffle=False)
   """

#MODELS:
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    # activation function
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # activation function

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.dropout(x, training=self.training)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.image_size = IMAGE_SIZE
        self.fc0 = nn.Linear(IMAGE_SIZE, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        b1 = self.batch_norm1d(self.fc0(x))
        x = f.relu(b1)
        b2 = self.batch_norm2d(self.fc1(x))
        x = f.relu(b2)
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

class ModelE(nn.Module):
    def __init__(self, input_size):
        super(ModelE, self).__init__()
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = f.relu(self.fc0(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)
        return f.log_softmax(x, dim=1)

class ModelF(nn.Module):
    def __init__(self, input_size):
        super(ModelF, self).__init__()
        self.image_size = input_size
        self.fc0 = nn.Linear(input_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, NUMBER_OF_LABELS)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return f.log_softmax(x, dim=1)


model = None
optimizer = None
if MODEL_TO_USE == 'A':
    model = ModelA()
elif MODEL_TO_USE == 'B':
    model = ModelB()
elif MODEL_TO_USE == 'C':
    model = ModelC()
elif MODEL_TO_USE == 'D':
    model = ModelD()
elif MODEL_TO_USE == 'E':
    model = ModelE()
elif MODEL_TO_USE == 'F':
     model = ModelF()

optimizer= model.optimizer
#so now we have our model
#we can start training

def train(epoch, model):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model.optimizer.step()
        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

    train_loss /= (len(train_loader.dataset) / 64)
    print('Train set, epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def validate(model, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)  # forward.
            # sum up batch loss and get the index of the max log-probability.
            validation_loss += f.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).cpu().sum()
    validation_loss /= (len(validation_loader) * 64)
    print('Validation Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, (len(validation_loader) * 64),
        100. * correct / (len(validation_loader) * 64)))

def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += f.nll_loss(output, target.type(torch.int64), reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log- probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set, epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
         epoch, test_loss, correct, len(test_loader.dataset),
         100. * correct / len(test_loader.dataset)))

def predict_y(model, test_x):
    model.eval()
    test_y_list = []
    for data in test_x:
        output = model(data)
        predict = output.max(1, keepdim=True)[1]
        test_y_list.append(str(int(predict)))
    return test_y_list

def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()


for epoch in range(EPOCHS):
    train(epoch,model)
    validate(model, validation_loader)



#for finish, we go to do the test and printing to file the results
prediction = predict_y(model, test_x_loader)
write_to_file(prediction)
