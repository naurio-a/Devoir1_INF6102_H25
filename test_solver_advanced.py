from utils import Node, Instance, Solution

from collections import defaultdict

import random
import math
import copy

from itertools import groupby, combinations

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, neighbors, label=0):
        super().__init__(idx, neighbors)
        self.__label = label

    def set_label(self, new_label) :
        self.__label = new_label

    def label(self) :
        return self.__label
    
    def __repr__(self) -> str:
        return 'Node'+str(['idx:'+str(self.get_idx()),'label:'+str(self.__label),'neighbors:'+str(self.neighbors())])
    

def get_label_dict(instance):
    """
    Creates a dictionnary key : label, value : list of nodes with this label
    """
    labels_dict = defaultdict(list)
    for node in instance.nodes:
        labels_dict[node.label()].append(node)
    return labels_dict

def modularity(instance):
    """
    Computes and returns the current modularity of the instance
    """
    Q = 0
    M2 = 2 * instance.M
    degree_dict = {node: node.degree() for node in instance.nodes}
    label_dict = get_label_dict(instance)

    for node1 in instance.nodes:
        label1 = node1.label()
        for node2 in label_dict.get(label1): 
            P = degree_dict[node1] * degree_dict[node2] / M2
            Q += (1 - P) if node2.get_idx() in node1.neighbors() else -(P)

    return Q / M2

def label_evaluation(instance, current_node, new_label, label_dict):
    # return terme de la somme à optimiser avec le nouveau label
    sum = 0
    for test_node in label_dict[new_label] :
        ### Réfléchir à pq c'est nécessaire
        if test_node == current_node :
            continue
        P = current_node.degree() * test_node.degree() / (2*instance.M)
        sum += (1 - P) if test_node.get_idx() in current_node.neighbors() else -(P)
    return sum

def update_label(instance, current_node) :

    label_dict = get_label_dict(instance)
    possible_labels = [i.label() for i in instance.nodes if (i.label() in current_node.neighbors())]
    possible_labels = list(set(possible_labels))
    current_label = current_node.label()

    # Determine best label for the node
    best_score = label_evaluation(instance, current_node, current_label, label_dict)
    best_labels = [current_label]
    #print(f'node {current_node.get_idx()}, possibles labels : {labels}, neighbors : {current_node.neighbors()}')
    
    for label in possible_labels :
        score = label_evaluation(instance, current_node, label, label_dict)
        #print(label, score, best_score)
        if score > best_score :
            best_score = score
            best_labels = [label]
        elif score == best_score :
            best_labels.append(label)
    new_label = random.choice(best_labels)

    return current_label, new_label

def LPAm(instance, dict_nodes, nodes_to_test) :

    Q = modularity(instance)
    i = 0
    step = 1
    # nodes_to_test = set(instance.nodes)
    #nodes_to_test2 = set()

    while i < len(nodes_to_test) :

        print(f'step LPAm : {step}, noeuds à tester : {len(nodes_to_test)}')
        # nodes_to_test2 = set()

        for node in nodes_to_test :
            #print(f'step {step} node {node.get_idx()}')
            current_label, new_label = update_label(instance, node)
            if current_label == new_label :
                i += 1
            else :
                node.set_label(new_label)
                # print(f'node {node}, new label : {new_label}')
                new_Q = modularity(instance)
                if (math.isclose(new_Q, Q, rel_tol=1e-7)) :
                    i += 1
                elif (new_Q > Q) :
                    i = 0
                else :
                    print("Erreur : la modularité a diminué")
                    return
                # for neighbor in dict_neighbors[node] :
                #     nodes_to_test2.add(dict_nodes[neighbor])
                Q = new_Q
         
        step += 1
        #nodes_to_test = nodes_to_test2


def merge_communities(instance, dict_nodes) :
    init_Q = modularity(instance)

    labels_dict = get_label_dict(instance)

    comb_Q_dict = {}

    # Get label duos
    label_duos = set()
    index_to_node = {node.get_idx(): node for nodes in labels_dict.values() for node in nodes}

    for label1, nodes in labels_dict.items():
        for node in nodes:
            for neighbor_idx in node.neighbors():
                neighbor = index_to_node.get(neighbor_idx)
                label2 = neighbor.label()
                if label1 != label2:
                    pair = tuple(sorted((label1, label2)))
                    label_duos.add(pair)

    # On teste les duos
    for duo in label_duos :
        original_labels = {i : i.label() for i in instance.nodes}

        # On merge deux communautés
        for node in instance.nodes :
            if node.label() == duo[0] :
                node.set_label(duo[1])

        # On regarde si le merge a amélioré Q
        duo_Q = modularity(instance)
        if duo_Q >= init_Q :
            comb_Q_dict[duo] = duo_Q

        # On reset pour continuer les tests
        for node in instance.nodes :
            node.set_label(original_labels[node])

    #### TEST
    if comb_Q_dict == {} :
        return False, {}
    
    else :
        nodes_to_test = set()
        sorted_merges = sorted(comb_Q_dict.items(), key=lambda x: x[1], reverse=True)
        merged_labels = set()

        # Apply non-conflicting merges
        for (label1, label2), _ in sorted_merges:
            if (label1 not in merged_labels) and (label2 not in merged_labels):
                for node in labels_dict[label1] :
                    node.set_label(label2)
                    for neighbor in node.neighbors() :
                        if neighbor.label() != label1 and neighbor.label != label2 :
                            nodes_to_test.add(dict_nodes[neighbor])
                merged_labels.add(label1)
                merged_labels.add(label2)
                #print(f'Communities merged : {label1} {label2}')

        return True, nodes_to_test

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """
    for i, node in enumerate(instance.nodes):
        custom_node = CustomNode(node.get_idx(), node.neighbors(), label=node.get_idx())
        instance.nodes[i] = custom_node

    dict_nodes = {i.get_idx() : i for i in instance.nodes}

    total_step = 1
    merge = True

    nodes_to_test = set(instance.nodes)

    while merge :
        print(total_step, modularity(instance))
        #print("début LPAM")
        LPAm(instance, dict_nodes, nodes_to_test)
        #print("fin LPAM, début merge")
        merge, nodes_to_test = merge_communities(instance, dict_nodes)
        #print("fin merge")
        total_step += 1

    sorted_nodes = sorted(instance.nodes, key=lambda node: node.label())
    groups = [list(map(lambda node: node.get_idx(), group)) for _, group in groupby(sorted_nodes, key=lambda node: node.label())]

    return Solution(groups)