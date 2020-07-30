import torch
import sys

from torch_geometric.datasets import TUDataset
from data.graph.preproc import Graph2TreesLoader
from torch_geometric.data import DataLoader
from graph_htn.graph_htn import GraphHTN
from cgmn.cgmn import CGMN
from sklearn.metrics import accuracy_score

def nci1_transform(data):
    data.x = data.x.argmax(1)
    data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
    return data

DEVICE=sys.argv[1]
N_GEN = int(sys.argv[2])
C = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
EPOCHS = int(sys.argv[5])

dataset = TUDataset('.', 'NCI1', transform=nci1_transform)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
cgmn = CGMN(1, N_GEN, C, 37, DEVICE)

bce = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(cgmn.parameters())
device = torch.device(DEVICE)
for i in range(EPOCHS):
    loss_avg = 0
    acc_avg = 0
    n = 0
    for b in loader:
        b = b.to(device)
        out, neg_likelihood = cgmn(b.x, b.edge_index, b.batch)
        loss = bce(out, b.y)
        opt.zero_grad()
        loss.backward()
        neg_likelihood.backward()
        opt.step()

        accuracy = accuracy_score(b.y.detach().cpu().numpy(), out.detach().cpu().sigmoid().numpy().round())
        loss_avg = loss.cpu().item() if n == 0 else loss_avg + ((loss.cpu().item() - loss_avg)/(n+1))
        acc_avg = accuracy if n == 0 else acc_avg + ((accuracy - acc_avg)/(n+1))
        n += 1
        print(f"Loss = {loss.item()} ----- Likelihood = {neg_likelihood.item()} ----- Accuracy = {accuracy}")

    print(f"---------- Loss avg at epoch {i} = {loss_avg} --  Accuracy = {acc_avg} -----------")
    if i > 0 and i%40 == 0:
        print(f"Appending Layer {len(cgmn.cgmm.layers)}")
        cgmn.stack_layer()
        opt = torch.optim.Adam(cgmn.parameters())
        
    
