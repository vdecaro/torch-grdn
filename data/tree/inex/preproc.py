import torch


INEX2005 = './data/tree/inex/2005/inex05.test.elastic.tree'
INEX2006 = './data/tree/inex/2006/inex06.test.elastic.tree'


def load_and_preproc_inex(file):
    with open(INEX2005 if file == 'inex2005' else INEX2006, "r") as ins:
        line_tree = []
        for line in ins:
            line_tree.append(line)
    ins.close()

    features = {'levels': [],
                'leaves': [],
                'labels': [],
                'pos': []}
    targets = []
    
    for line in line_tree:
        edges, leaves, labels, pos, target = _build_tree(line)
        features['levels'].append(edges)
        features['leaves'].append(leaves)
        features['labels'].append(labels)
        features['pos'].append(pos)

        targets.append(target)

    return features, targets


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

    edges = [torch.tensor(l) for l in edges]
    labels = torch.tensor(labels)
    leaves = torch.tensor(leaves)
    pos = torch.tensor([0]+pos)
    return edges, leaves, labels, pos, int(t_class)-1