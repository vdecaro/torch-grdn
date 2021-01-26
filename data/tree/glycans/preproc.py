import os
import torch
from torch_geometric.data import Data


LEUKEMIA = 'data/tree/glycans/leukemia/trees'
CYSTIC = 'data/tree/glycans/cystic/trees'

def load_and_preproc_glycans(work_dir, name):
    if name == 'leukemia':
        f = LEUKEMIA
    if name == 'cystic':
        f = CYSTIC

    with open(os.path.join(work_dir, f), "r") as ins:
        t_lines = []
        for line in ins:
            t_lines.append(line)
    
    data = _build_trees(t_lines)
    return data


def _build_trees(t_lines):
    labels_map = {}
    n_labels = 0
    data = []
    for i, line in enumerate(t_lines):
        tab_split = line.split('\t')
        t_class, t_line = tab_split[0], tab_split[2]
        t_line = t_line.replace(" ", "").replace("$-", "$")
        labels = []
        pos = []
        edges = []
        leaves = []

        stack = []
        curr_label = ''
        popping = False
        for c in t_line:
            if c == '(':
                if curr_label != '':
                    level = len(stack)-1
                    if len(edges) == level:
                        edges.append([])
                    
                    if stack:
                        edges[level].append([stack[-1][0], len(labels)])
                        pos.append(stack[-1][1])
                        stack[-1][1] += 1

                    stack.append([len(labels), 0])
                    if not curr_label in labels_map:
                        labels_map[curr_label] = n_labels
                        n_labels += 1
                    labels.append(labels_map[curr_label])

                    curr_label = ''
                popping = False
            elif c == '$':
                leaves.append(stack[-1][0])
            elif c == ')':
                if popping:
                    stack.pop()
                popping = True
            else:
                curr_label += c

        edges = [torch.LongTensor(l).T for l in edges]
        labels = torch.LongTensor(labels)
        leaves = torch.LongTensor(leaves)
        pos = torch.LongTensor([0]+pos)
        t_class = 0 if int(t_class) == -1 else 1

        data.append(Data(levels=edges, leaves=leaves, x=labels, pos=pos, y=[t_class], dim=labels.size(0)))
        
    return data
