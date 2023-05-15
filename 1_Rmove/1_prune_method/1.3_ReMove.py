from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel,self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,16,kernel_size=3,padding=1)
        self.fc=nn.Linear(16*28*28,10)
        
        self.conv1_l1norm = nn.Parameter(torch.Tensor(32), requires_grad=False)
        self.conv2_l1norm = nn.Parameter(torch.Tensor(32), requires_grad=False)
        self.register_buffer('conv1_l1norm_buffer', self.conv1_l1norm)
        self.register_buffer('conv2_l1norm_buffer', self.conv2_l1norm)
    def forward(self,x):
        x=self.conv1(x)
        #output [32,1,3,3]
        self.conv1_l1norm.data=torch.sum(torch.abs(self.conv1.weight.data),dim=(1,2,3))
        x=self.conv2(x)
        self.conv2_l1norm.data=torch.sum(torch.abs(self.conv2.weight.data),dim=(0,2,3))
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
def train(model, dataloader, criterion, optimizer, device='cpu', num_epochs=10):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(f"Loss: {running_loss / len(dataloader)}")           
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    return model
if __name__=='__main__':
    # Prepare the MNIST dataset
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    big_model=BigModel()
    num_epochs=3
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(big_model.parameters(),lr=1e-3)
    big_model=train(big_model, train_loader, criterion, optimizer, device='cuda', num_epochs=3)
    torch.save(big_model.state_dict(),"big_model.pth")

    #export onnx
    dummy_input=torch.randn(1,1,28,28).to('cuda')
    torch.onnx.export(big_model,dummy_input,"big_model.onnx")