# FedWM
Code for the paper "Ownership Privacy-Preserving DNN Watermark for Federated Learning".
PyTorch>=1.11
CUDA required

## File structure
1. ./data : Where the dataset is stored by default.
2. LeNet.py : The demo implementation for FedWM on LeNet with MNIST. 
3. AUC.py : The AUC function.

To run the demo of FedWM, run 

`python FedWM.py`

The settings of the experiments can be modified through parameters within the script:
```python
# The number of participants.
K=80
# The number of influenced parameters.
r=0.1
# The scale of communities.
Asize=10
# The number of FL training epochs.
E=20
```

Then major metric of interest, owership verification AUC is computed through:
```python
negs=[]
for a in Authors:
    negs.append(a.uc.connect(FCNN1))
poss=[]
for i in range(K):
    u=UCVerifier(r)
    poss.append(u.connect(FCNN1))
print("AUC=",end="")
print(AUC(negs,poss))
```

To generalize the proposed scheme to other networks/datasets, modifying the settings of the network/dataset initialization in sufficient. 

