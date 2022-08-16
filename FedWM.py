import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import torch.utils.data as Data
import numpy as np
import copy
import torch.nn.utils.prune as prune
import time
import random
from AUC import AUC

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=(5,5))
        self.relu1=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=(5,5))
        self.relu2=nn.ReLU()
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv3=nn.Conv2d(16,120,kernel_size=(5,5))
        self.relu3=nn.ReLU()
        self.fc1=nn.Linear(120,84)
        self.relu4=nn.ReLU()
        self.fc2=nn.Linear(84,10)
    def forward(self,img,out_feature=False):
        output=self.conv1(img)
        output=self.relu1(output)
        output=self.maxpool1(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.maxpool2(output)
        output=self.conv3(output)
        output=self.relu3(output)
        feature=output.view(-1, 120)
        output=self.fc1(feature)
        output=self.relu4(output)
        output=self.fc2(output)
        if out_feature==False:
            return output
        else:
            return output,feature

def train(CNN,train_loader,lr,device,verbose=True):
    CNN=CNN.to(device)
    for param in CNN.parameters():
        param.requires_grad=True
    optimizer=optim.Adam(CNN.parameters(),lr=lr)
    E=1
    lam=0.05
    for epoch in range(E):
        for idx,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            optimizer.zero_grad()
            pred=CNN(b_x)
            l=F.cross_entropy(pred,b_y)
            l.backward()
            optimizer.step()
            if idx%100==0 and verbose:
                print("Epoch=%i,Index=%i,Loss=%f"%(epoch,idx,float(l.detach())/bs))
    return True

def test(CNN,test_loader,device,verbose=True):
    CNN.eval()
    CNN=CNN.to(device)
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=CNN(data)
            test_loss+=F.cross_entropy(output,target,reduction='sum').item() # sum up batch loss
            pred=output.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss,correct,len(test_loader.dataset),
        100. * correct/len(test_loader.dataset)))
    return test_loss

class UCVerifier(nn.Module):
    def __init__(self,r=0.1):
        super(UCVerifier,self).__init__()
        self.r=r
        self.mask1=torch.zeros(size=(16,6,5,5))
        self.mask2=torch.zeros(size=(120,16,5,5))
        self.mask1.requires_grad=False
        self.mask2.requires_grad=False
        for i in range(16):
            for j in range(6):
                for k in range(5):
                    for l in range(5):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask1[i][j][k][l]=1
        for i in range(120):
            for j in range(16):
                for k in range(5):
                    for l in range(5):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask2[i][j][k][l]=1
        self.carrier1=torch.randn(size=(16,6,5,5))
        self.carrier2=torch.randn(size=(120,16,5,5))
    def connect(self,CNN):
        c1=CNN.state_dict()["conv2.weight"].detach().cpu()
        c2=CNN.state_dict()["conv3.weight"].detach().cpu()
        e1=(c1-self.carrier1)*self.mask1
        e2=(c2-self.carrier2)*self.mask2
        if self.r==0:
            return 0
        else:
            return float(torch.sum(e1**2)+torch.sum(e2**2))/(16*6*5*5+120*16*5*5)/self.r
    def fit(self,CNN,D,Dk):
        c1=CNN.state_dict()["conv2.weight"].detach().cpu()
        c2=CNN.state_dict()["conv3.weight"].detach().cpu()
        for i in range(16):
            for j in range(6):
                for k in range(5):
                    for l in range(5):
                        if self.mask1[i][j][k][l]==1:
                            c1[i][j][k][l]=(self.carrier1[i][j][k][l]*D-c1[i][j][k][l]*(D-Dk))/Dk
        for i in range(120):
            for j in range(16):
                for k in range(5):
                    for l in range(5):
                        if self.mask2[i][j][k][l]==1:
                            c2[i][j][k][l]=(self.carrier2[i][j][k][l]*D-c2[i][j][k][l]*(D-Dk))/Dk
        d=CNN.state_dict()
        d["conv2.weight"]=c1
        d["conv3.weight"]=c2
        CNN.load_state_dict(d)
        return True

#--------------------------------------------------------------------------------------------------------------------------------------------

bs=128

data_root="./data"
train_loader=torch.utils.data.DataLoader( 
            datasets.MNIST(data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=bs, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32,32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=bs,shuffle=True,num_workers=2)

print("MNIST Data loaded.")

#--------------------------------------------------------------------------------------------------------------------------------------------

dataset=datasets.MNIST(data_root,train=True,download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32,32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ]))

def CNN_minus(CNN1,CNN2):
    if CNN1.state_dict().keys()!=CNN2.state_dict().keys():
        return False
    diff={}
    CNN1=CNN1.to(torch.device("cpu"))
    CNN2=CNN2.to(torch.device("cpu"))
    for key in CNN1.state_dict().keys():
        diff[key]=CNN1.state_dict()[key]-CNN2.state_dict()[key]
        diff[key]=diff[key]
    diffCNN=MyLeNet()
    diffCNN.load_state_dict(diff)
    return diffCNN

def CNN_plus(CNN1,CNN2,f):
    if CNN1.state_dict().keys()!=CNN2.state_dict().keys():
        return False
    plus={}
    CNN1=CNN1.to(torch.device("cpu"))
    CNN2=CNN2.to(torch.device("cpu"))
    for key in CNN1.state_dict().keys():
        plus[key]=CNN1.state_dict()[key]+CNN2.state_dict()[key]*f
    plusCNN=MyLeNet()
    plusCNN.load_state_dict(plus)
    return plusCNN

# For poisoning detection.
"""
def CNN_cosd(CNN1,CNN2):
    if CNN1.state_dict().keys()!=CNN2.state_dict().keys():
        return False
    plus={}
    CNN1=CNN1.to(torch.device("cpu"))
    CNN2=CNN2.to(torch.device("cpu"))
    cosds=[]
    for key in CNN1.state_dict().keys():
        t=torch.sum(CNN1.state_dict()[key]*CNN2.state_dict()[key])/torch.sqrt(torch.sum(CNN1.state_dict()[key]*CNN1.state_dict()[key]))/torch.sqrt(torch.sum(CNN2.state_dict()[key]*CNN2.state_dict()[key]))
        cosds.append(t)
    return cosds

def CNN_l2(CNN1,CNN2):
    if CNN1.state_dict().keys()!=CNN2.state_dict().keys():
        return False
    plus={}
    CNN1=CNN1.to(torch.device("cpu"))
    CNN2=CNN2.to(torch.device("cpu"))
    l2ds=[]
    for key in CNN1.state_dict().keys():
        t=torch.sum((CNN1.state_dict()[key]-CNN2.state_dict()[key])**2)
        l2ds.append(t)
    s=0
    for i in l2ds:
        s=s+i
    return s
"""

class LocalDataset(Dataset):
    def __init__(self,dataset,mask):
        self.dataset=dataset
        self.mask=mask
    def __getitem__(self,index):
        return self.dataset[self.mask[index]]
    def __len__(self):
        return len(self.mask)

class Author(object):
    def __init__(self,dataset,mask,r):
        self.localdataset=LocalDataset(dataset,mask)
        if len(self.localdataset)>=128:
            self.bs=128
        else:
            self.bs=len(self.localdataset)
        self.uc=UCVerifier(r)
        self.auc=UCVerifier(0.0)
        self.inA=False
    def update(self,CNN,device):
        CNNt=CNN.to(torch.device("cpu"))
        localCNN=copy.deepcopy(CNN)
        local_train_loader=torch.utils.data.DataLoader(self.localdataset,batch_size=bs,shuffle=True,num_workers=2)
        train(localCNN,local_train_loader,0.0005,device,False)
        localCNN=localCNN.to(torch.device("cpu"))
        return localCNN,CNN_minus(localCNN,CNNt)
    def IP_UC_update(self,CNN,device):
        CNNt=CNN.to(torch.device("cpu"))
        localCNN,_=self.update(CNN,device)
        self.uc.fit(localCNN)
        if self.inA:
            self.auc.fit(localCNN,len(self),60000)
        return localCNN,CNN_minus(localCNN,CNNt)
    def __len__(self):
        return len(self.localdataset)

def FedAvg(CNN,device,Authors):
    N=0
    CNN=CNN.to(torch.device("cpu"))
    tempCNN=copy.deepcopy(CNN)
    for author in Authors:
        N=N+len(author)
    for author in Authors:
        _,gradient=author.IP_UC_update(tempCNN,device)
        f=len(author)/N
        CNN=CNN_plus(CNN,gradient,f)
    return CNN

def FedWM(CNN,device,Authors,A):
    for a in Authors:
        a.inA=False
    for a in A:
        a.inA=True
    A[0].auc=UCVerifier(0)
    for i in range(len(A)):
        if i!=0:
            a.auc=UCVerifier(0.01)
            A[0].auc.mask1=A[0].auc.mask1+a.auc.mask1
            A[0].auc.mask2=A[0].auc.mask2+a.auc.mask2
            for i in range(16):
                for j in range(6):
                    for k in range(5):
                        for l in range(5):
                            if A[0].auc.mask1[i][j][k][l]!=0:
                                A[0].auc.mask1[i][j][k][l]=1
                                A[0].auc.carrier1[i][j][k][l]=CNN.state_dict()["conv2.weight"][i][j][k][l]/len(A[0])*60000-a.auc.carrier1[i][j][k][l]
            for i in range(120):
                for j in range(16):
                    for k in range(5):
                        for l in range(5):
                            if A[0].auc.mask2[i][j][k][l]!=0:
                                A[0].auc.mask2[i][j][k][l]=1
                                A[0].auc.carrier2[i][j][k][l]=CNN.state_dict()["conv3.weight"][i][j][k][l]/len(A[0])*60000-a.auc.carrier2[i][j][k][l]
    CNN=FedAvg(CNN,device,Authors)
    for a in Authors:
        a.inA=False
    return CNN
           
FCNN=MyLeNet()

K=80
r=0.1
device=torch.device("cuda:3")
step=int(60000/K)
Authors=[]
for i in range(K):
    a=Author(dataset,list(range(step*i,step*(i+1))),r)
    Authors.append(a)
Asize=10
E=20
losscurve=[]
for epoch in range(E):
    print("Epoch %i in %i"%(epoch+1,E))
    A=random.sample(Authors,Asize)
    start_time=time.time()
    FCNN1=FedWM(FCNN1,device,Authors,A)
    print("Time elapsed=%f"%(time.time()-start_time))
    l=test(FCNN1,test_loader,device)
    losscurve.append(l)

negs=[]
for a in Authors:
    negs.append(a.uc.connect(FCNN1))
poss=[]
for i in range(K):
    u=UCVerifier(r)
    poss.append(u.connect(FCNN1))
print("AUC=",end="")
print(AUC(negs,poss))

