import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset



INEX2005 = './data/graph/inex/2005/inex05'
INEX2006 = './data/graph/inex/2006/inex06'

def load_and_preproc_inex(file):
    if file == 'inex2005train':
        f = f'{INEX2005}.train.elastic.tree'
    if file == 'inex2005test':
        f = f'{INEX2005}.test.elastic.tree'
    if file == 'inex2006train':
        f = f'{INEX2006}.train.elastic.tree'
    if file == 'inex2006test':
        f = f'{INEX2006}.test.elastic.tree'
    with open(f, "r") as ins:
        line_tree = []
        for line in ins:
            line_tree.append(line)
    
    data = [_build_graph(line) for line in line_tree]
    return data


def _build_graph(line):
    t_class, t_line = line.split(':')
    labels = []
    edges = []
    pos = []
    t_iter = iter(t_line)
    c = next(t_iter)

    stack = []
    curr_label = ''
    while True:
        try:
            if c == '(':
                if stack:
                    edges.append([stack[-1][0], len(labels)])
                    edges.append([len(labels), stack[-1][0]])
                    pos.append(stack[-1][1])
                    pos.append(stack[-1][1])
                    stack[-1][1] += 1
                    
                stack.append([len(labels), 0])
                labels.append(int(curr_label)-1)
                curr_label = ''

            elif c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                curr_label += c
            elif c == '$':
                pass
            elif c == ')':
                stack.pop()
            c = next(t_iter)
        except StopIteration:
            break

    edges = torch.LongTensor(edges).T
    x = torch.LongTensor(labels)
    y = torch.LongTensor([int(t_class)-1])
    pos = torch.LongTensor(pos)
    return Data(edge_index=edges, x=x,y=y, pos=pos)