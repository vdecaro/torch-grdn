import torch
from torch_geometric.data import Data


INEX2005 = './data/tree/inex/2005/inex05'
INEX2006 = './data/tree/inex/2006/inex06'


def load_and_preproc_inex(file):
    if file == 'inex2005train':
        f = '{}.train.elastic.tree'.format(INEX2005)
    if file == 'inex2005test':
        f = '{}.test.elastic.tree'.format(INEX2005)
    if file == 'inex2006train':
        f = '{}.train.elastic.tree'.format(INEX2006)
    if file == 'inex2006test':
        f = '{}.test.elastic.tree'.format(INEX2006)
    with open(f, "r") as ins:
        line_tree = []
        for line in ins:
            line_tree.append(line)
    
    data = [_build_tree(line) for line in line_tree]
    return data


def _build_tree(line):
    t_class, t_line = line.split(':')
    labels = []
    pos = []
    edges = []
    leaves = []
    t_iter = iter(t_line)
    c = next(t_iter)

    stack = []
    curr_label = ''
    while True:
        try:
            if c == '(':
                level = len(stack)-1
                if len(edges) == level:
                    edges.append([])
                if stack:
                    edges[level].append([stack[-1][0], len(labels)])
                    pos.append(stack[-1][1])
                    stack[-1][1] += 1
                    
                stack.append([len(labels), 0])
                labels.append(int(curr_label)-1)
                curr_label = ''

            elif c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                curr_label += c
            elif c == '$':
                leaves.append(stack[-1][0])
            elif c == ')':
                stack.pop()
            c = next(t_iter)
        except StopIteration:
            break

    edges = [torch.LongTensor(l).T for l in edges]
    labels = torch.LongTensor(labels)
    leaves = torch.LongTensor(leaves)
    pos = torch.LongTensor([0]+pos)
    dim = torch.LongTensor([labels.size(0)])
    y = torch.LongTensor([int(t_class)-1])
    return Data(levels=edges, leaves=leaves, x=labels, pos=pos, y=y, dim=dim)