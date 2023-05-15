from ReMove import BigModel, train
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
    
if __name__=='__main__':
    ##  1.Prune local threshold
    model = BigModel()
    model.load_state_dict(torch.load("E:\\Study\\4_Git\\learn-prune\\workspace\\big_model.pth"))
    
    conv1=model.conv1
    conv2=model.conv2
    conv1_l1norm=model.conv1_l1norm
    conv2_l1norm=model.conv2_l1norm

    length=int(conv1_l1norm.data.size(0)*0.5)
    threshold=torch.sort(conv1_l1norm.data)[0][length]
    ## Top conv
    keep_idxs=torch.where(conv1_l1norm >= threshold)[0]
    # print(keep_idxs)
    conv1.weight.data=conv1.weight.data[keep_idxs]
    conv1.bias.data=conv1.bias.data[keep_idxs]
    conv1_l1norm.data = conv1_l1norm.data[keep_idxs]
    conv1.out_channels = length
    ##  Bottom conv
    # print(conv2_l1norm)
    _, keep_idxs2=torch.topk(conv2_l1norm, length)
    print(keep_idxs2)
    conv2.weight.data = conv2.weight.data[:,keep_idxs2]
    conv2.in_channels = length

    torch.save(model.state_dict(), "E:\\Study\\4_Git\\learn-prune\\workspace\\mypruned_model.pth")
    # dummy_input=torch.randn(1,1,28,28)
    # torch.onnx.export(model, dummy_input, "mypruned_model.onnx")

    
#################################### FINE TUNE #####################################
    # Prepare the MNIST dataset
    model.load_state_dict(torch.load("E:\\Study\\4_Git\\learn-prune\\workspace\\mypruned_model.pth"))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    big_model = train(model, train_loader, criterion, optimizer, device='cuda', num_epochs=3)

    torch.save(big_model.state_dict(), "pruned_model_after_finetune.pth")
