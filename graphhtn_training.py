import torch

from torch_geometric.datasets import TUDataset
from data.graph.preproc import Graph2TreesLoader
from torch_geometric.data import DataLoader
from graph_htn.graph_htn import GraphHTN
from cgmn.cgmn import CGMN
from sklearn.metrics import accuracy_score

torch.autograd.set_detect_anomaly(True)

def nci1_transform(data):
    data.x = data.x.argmax(1)
    data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
    return data

dataset = TUDataset('.', 'NCI1', transform=nci1_transform)

loader = Graph2TreesLoader(dataset, max_depth=4, batch_size=64, shuffle=True)
C = 4
ghtn = GraphHTN(1, 20, 0, C, 37)
bce = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(ghtn.get_parameters())

for i in range(200):
    loss_avg = 0
    acc_avg = 0
    n = 0
    for b in loader:
        out, neg_likelihood = ghtn(b.x, b.trees, b.batch)
        loss = bce(out, b.y)
        opt.zero_grad()
        loss.backward()
        neg_likelihood.backward()
        opt.step()

        accuracy = accuracy_score(b.y.detach().numpy(), out.detach().sigmoid().numpy().round())
        loss_avg = loss.item() if n == 0 else loss_avg + ((loss.item() - loss_avg)/(n+1))
        acc_avg = accuracy if n == 0 else acc_avg + ((accuracy - acc_avg)/(n+1))
        n += 1
        print(f"Loss = {loss.item()} ----- Likelihood = {neg_likelihood.item()} ----- Accuracy = {accuracy}")

    print(f"---------- Loss avg at epoch {i} = {loss_avg} --  Accuracy = {acc_avg} -----------")