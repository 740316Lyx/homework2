import torch
import pickle
import torch.nn as nn
from sklearn import preprocessing
from torch.nn import Parameter, Module
import torch.nn.functional as F
import torch.optim as optim
import pprint, copy, os, random, math, sys, pickle, time
import numpy as np
import networkx as nx
from multiprocessing import Process, Pool

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
verb_net3_mapping_with_args = '../data/encoding_with_args.csv'

def trans_to_cuda(variable):
    if use_cuda:
        return variable.cuda()
    else:
        return variable

def id_to_vec(emb_file):
    dic = {}
    for s in open(emb_file):
        s = s.strip().split()
        if len(s) == 2:
            continue
        dic[s[0]] = np.array(s[1:], dtype=np.float32)
    dic['0'] = np.zeros(len(dic['0']), dtype=np.float32)
    return dic

def word_to_id(voc_file):
    dic = {}
    for s in open(voc_file):
        s = s.strip().split()
        dic[s[1]] = s[0]
    return dic

def get_word_vec(id_vec):
    word_vec = []
    for i in range(len(id_vec)):
        word_vec.append(id_vec[str(i)])
    return np.array(word_vec, dtype=np.float32)

def get_hash_for_word(emb_file, voc_file):
    id_vec = id_to_vec(emb_file)
    return word_to_id(voc_file), id_vec, get_word_vec(id_vec)

class Data_data(object):
    def __init__(self, questions, questions2=None):
        super(Data_data, self).__init__()
        if questions2 is None:
            self.A, self.input_data, self.targets = questions[0], questions[1], questions[2]
        else:
            self.A = torch.cat((questions[0], questions2[0]))
            self.input_data = torch.cat((questions[1], questions2[1]))
            self.targets = torch.cat((questions[2], questions2[2]))
        self.corpus_length = len(self.targets)
        self.start = 0

    def next_batch(self, batch_size):
        start = self.start
        end = (self.start + batch_size) if (self.start + batch_size) <= self.corpus_length else self.corpus_length
        self.start = (self.start + batch_size)
        if self.start < self.corpus_length:
            epoch_flag = False
        else:
            self.start = self.start % self.corpus_length
            epoch_flag = True
        return [trans_to_cuda(self.A[start:end]), trans_to_cuda(self.input_data[start:end]), trans_to_cuda(self.targets[start:end])], epoch_flag

    def all_data(self, index=None):
        if index is None:
            return [trans_to_cuda(self.A), trans_to_cuda(self.input_data), trans_to_cuda(self.targets)]
        else:
            return [trans_to_cuda(self.A.index_select(0, index)), trans_to_cuda(self.input_data.index_select(0, index)), trans_to_cuda(self.targets.index_select(0, index))]

def get_event_chains(event_list):
    return [['%s_%s' % (ev[0], ev[2]) for ev in event_list], ['%s' % ev[3] for ev in event_list], ['%s' % ev[4] for ev in event_list], ['%s' % ev[5] for ev in event_list]]

class Data_txt(object):
    def __init__(self, questions):
        super(Data_txt, self).__init__()
        self.corpus = questions
        self.corpus_length = len(questions)
        self.start = 0

    def next_batch(self, batch_size):
        batch = []
        for i in range(self.start, self.start + batch_size):
            i = i % self.corpus_length
            q = self.corpus[i]
            context_chains = get_event_chains(q[0])
            choices_chains = get_event_chains(q[1])
            batch.append([context_chains, choices_chains, q[2]])
        self.start = (self.start + batch_size)
        if self.start < self.corpus_length:
            epoch_flag = False
        else:
            self.start = self.start % self.corpus_length
            epoch_flag = True
        return batch, epoch_flag

    def all_data(self):
        batch = []
        for i in range(0, self.corpus_length):
            q = self.corpus[i]
            context_chains = get_event_chains(q[0])
            choices_chains = get_event_chains(q[1])
            batch.append([context_chains, choices_chains, q[2]])
        return batch

def build_graph(filename):
    graph = nx.DiGraph()
    for s in open(filename):
        s = s.strip().split()
        graph.add_edge(s[0], s[1], weight=float(s[2]))
    return graph

def return_id_list(event_list, word_id):
    id_list = []
    for event in event_list:
        if event in word_id:
            id_list.append(word_id[event])
        else:
            id_list.append('0')
    return id_list

def get_matrix(g, node_list, edge_list):
    node_list_map = {}
    for i in node_list:
        node_list_map[i] = len(node_list_map)
    node_num = len(node_list)
    A = np.zeros((node_num, node_num), dtype=np.float32)
    for edge in edge_list:
        start = edge[0]
        end = edge[1]
        A[node_list_map[start]][node_list_map[end]] = g[start][end]['weight']
    return A

def get_matrix_for_chain(g, node_list, edge_list):
    node_list_map = {}
    for i in node_list:
        node_list_map[i] = len(node_list_map)
    node_num = len(node_list)
    A = np.zeros((node_num, node_num), dtype=np.float32)
    for i, node in enumerate(node_list[0:7]):
        if (node_list[i], node_list[i+1]) in edge_list:
            start = node_list[i]
            end = node_list[i+1]
            A[node_list_map[start]][node_list_map[end]] = g[start][end]['weight']
    for i, node in enumerate(node_list[8:13]):
        if (node_list[7], node) in edge_list:
            start = node_list[7]
            end = node
            A[node_list_map[start]][node_list_map[end]] = g[start][end]['weight']
    return A

def process(data, word_id, g, predict=False):
    input_data = []
    targets = []
    A = []
    pbar = get_progress_bar(len(data), title='Process Data')
    for i in range(len(data)):
        pbar.update(i)
        context, choice, answer = data[i]
        targets.append(answer)
        context_id = return_id_list(context[0], word_id)
        choice_id = return_id_list(choice[0], word_id)
        context_subject_id = return_id_list(context[1], word_id)
        choice_subject_id = return_id_list(choice[1], word_id)
        context_object_id = return_id_list(context[2], word_id)
        choice_object_id = return_id_list(choice[2], word_id)
        context_perp_id = return_id_list(context[3], word_id)
        choice_perp_id = return_id_list(choice[3], word_id)
        node_list = context_id + choice_id
        node_list_int = [int(i) for i in node_list]
        node_list_subject = context_subject_id + choice_subject_id
        node_list_int_subject = [int(i) for i in node_list_subject]
        node_list_object = context_object_id + choice_object_id
        node_list_int_object = [int(i) for i in node_list_object]
        node_list_perp = context_perp_id + choice_perp_id
        node_list_int_perp = [int(i) for i in node_list_perp]
        input_data.append(node_list_int + node_list_int_subject + node_list_int_object + node_list_int_perp)
        new_g = g.subgraph(node_list)
        edge_list = list(new_g.edges())
        A.append(get_matrix_for_chain(new_g, node_list, edge_list))
    pbar.finish()
    A = torch.from_numpy(np.array(A))
    if not predict:
        input_data = torch.from_numpy(np.array(input_data))
    else:
        input_data = torch.from_numpy(np.array(input_data))
    targets = torch.from_numpy(np.array(targets))
    return A, input_data, targets

def dump_data():
    dev_small_data = Data_txt(pickle.load(open('../data/corpus_index_dev_small.txt', 'rb')))
    dev_data = Data_txt(pickle.load(open('../data/corpus_index_dev.txt', 'rb')))
    test_data = Data_txt(pickle.load(open('../data/corpus_index_test.txt', 'rb')))
    train_data = Data_txt(pickle.load(open('../data/corpus_index_train0.txt', 'rb')))
    print('train data prepare done')
    word_id, id_vec, word_vec = get_hash_for_word('../data/deepwalk_128_unweighted_with_args.txt', verb_net3_mapping_with_args)
    g = build_graph('../data/data2.csv')
    print('word vector prepare done')
    A, input_data, targets = process(dev_small_data.all_data(), word_id, g)
    pickle.dump([A, input_data, targets], open('../data/corpus_index_dev_small_with_args_all_chain.data', 'wb'), -1)
    print('dev_small_data done.')
    A, input_data, targets = process(dev_data.all_data(), word_id, g)
    pickle.dump([A, input_data, targets], open('../data/corpus_index_dev_with_args_all_chain.data', 'wb'), -1)
    print('dev_data done.')
    A, input_data, targets = process(test_data.all_data(), word_id, g)
    pickle.dump([A, input_data, targets], open('../data/corpus_index_test_with_args_all_chain.data', 'wb'), -1)
    print('test_data done.')
    A, input_data, targets = process(train_data.all_data(), word_id, g)
    pickle.dump([A, input_data, targets], open('../data/corpus_index_train0_with_args_all_chain.data', 'wb'), -1)
    print('train_data done.')

def process_matrix(data):
    A = data[0]
    new_A = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                if A[i, j, k] != 0:
                    new_A[i, j, k] = 0.01
    return [new_A, data[1], data[2]]

def change_graph_to_unweighted():
    dev_data_small = pickle.load(open('../data/corpus_index_dev_small_with_args_all.data', 'rb'))
    dev_data = pickle.load(open('../data/corpus_index_dev_with_args_all.data', 'rb'))
    test_data = pickle.load(open('../data/corpus_index_test_with_args_all.data', 'rb'))
    train_data = pickle.load(open('../data/corpus_index_train0_with_args_all.data', 'rb'))
    pickle.dump(process_matrix(dev_data_small), open('../data/corpus_index_dev_small_with_args_all_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(dev_data), open('../data/corpus_index_dev_with_args_all_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(test_data), open('../data/corpus_index_test_with_args_all_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(train_data), open('../data/corpus_index_train0_with_args_all_unweighted.data', 'wb'), -1)

def change_chain_to_unweighted():
    dev_data_small = pickle.load(open('../data/corpus_index_dev_small_with_args_all_chain.data', 'rb'))
    dev_data = pickle.load(open('../data/corpus_index_dev_with_args_all_chain.data', 'rb'))
    test_data = pickle.load(open('../data/corpus_index_test_with_args_all_chain.data', 'rb'))
    train_data = pickle.load(open('../data/corpus_index_train0_with_args_all_chain.data', 'rb'))
    pickle.dump(process_matrix(dev_data_small), open('../data/corpus_index_dev_small_with_args_all_chain_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(dev_data), open('../data/corpus_index_dev_with_args_all_chain_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(test_data), open('../data/corpus_index_test_with_args_all_chain_unweighted.data', 'wb'), -1)
    pickle.dump(process_matrix(train_data), open('../data/corpus_index_train0_with_args_all_chain_unweighted.data', 'wb'), -1)
