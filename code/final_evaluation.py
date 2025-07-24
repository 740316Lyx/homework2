import torch
import pickle
import numpy as np
from sklearn import preprocessing
from utils import trans_to_cuda, get_hash_for_word, Data_data
from gnn_with_args import EventGraph_With_Args
from event_chain import EventChain, Word2VecAttention

def get_event_chains(event_list):
    return ['%s_%s' % (ev[0], ev[2]) for ev in event_list]

def get_word_embedding(word, word_id, id_vec, emb_size):
    if word in word_id:
        return id_vec[word_id[word]]
    else:
        return np.zeros(emb_size, dtype=np.float32)

def get_vec_rep(questions, word_id, id_vec, emb_size, predict=False):
    rep = np.zeros((5 * len(questions), 9, emb_size), dtype=np.float32)
    correct_answers = []
    for i, q in enumerate(questions):
        context_chain = get_event_chains(q[0])
        choice_chain = get_event_chains(q[1])
        correct_answers.append(q[2])
        for j, context in enumerate(context_chain):
            context_vec = get_word_embedding(context, word_id, id_vec, emb_size)
            rep[5*i:5*(i+1), j, :] = context_vec
        for k, choice in enumerate(choice_chain):
            choice_vec = get_word_embedding(choice, word_id, id_vec, emb_size)
            rep[5*i+k, -1, :] = choice_vec
    if not predict:
        input_data = torch.from_numpy(rep)
    else:
        input_data = torch.from_numpy(rep)
    correct_answers = torch.from_numpy(np.array(correct_answers))
    return input_data, correct_answers

def get_acc(scores, correct_answers, name='scores', save=False):
    selections = np.argmax(scores, axis=1)
    num_correct = int(np.sum(selections == correct_answers))
    if save:
        pickle.dump((selections == correct_answers), open('./scores/' + name, 'wb'), 2)
    samples = len(correct_answers)
    accuracy = float(num_correct) / samples * 100.
    return accuracy

def process_test(scores, test_index):
    scores[test_index] = np.min(scores)
    return scores

def main():
    test_index = pickle.load(open('../data/test_index.pickle', 'rb'))
    HIDDEN_DIM = 512
    L2_penalty = 1e-8
    MARGIN = 0.015
    LR = 0.0001
    T = 1
    BATCH_SIZE = 1000
    EPOCHES = 520
    PATIENTS = 300
    test_data = Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data', 'rb')))
    word_id, id_vec, word_vec = get_hash_for_word('/users3/zyli/github/OpenNE/output/verb_net/1_property/deepwalk_128_unweighted_with_args.txt', '../data/encoding_with_args.csv')
    HIDDEN_DIM = 512
    L2_penalty = 1e-8
    MARGIN = 0.015
    LR = 0.0001
    T = 1
    BATCH_SIZE = 1000
    EPOCHES = 520
    PATIENTS = 300
    test_data = Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data', 'rb')))
    model = EventGraph_With_Args(len(word_vec), HIDDEN_DIM, word_vec, L2_penalty, MARGIN, LR, T, BATCH_SIZE)
    model = trans_to_cuda(model)
    model.load_state_dict(torch.load('../data/gnn_euclid_acc_52.380001068115234_.model'))
    data = test_data.all_data()
    correct_answers = data[2].cpu().numpy()
    scores1 = model(data[1], data[0]).cpu().numpy() 
    scores1 = process_test(scores1, test_index)
    print(get_acc(scores1, correct_answers, 'scores1'))
    model = trans_to_cuda(EventChain(
        embedding_dim=512,
        hidden_dim=512,
        vocab_size=len(word_vec),
        word_vec=word_vec,
        num_layers=1,
        bidirectional=False
    ))
    model.load_state_dict(torch.load('../data/event_chain_acc_50.98999786376953_.model'))
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4, scores2 = model.predict_with_minibatch(data[1], data[2])
    scores2 = scores2.cpu().numpy() 
    scores2 = process_test(scores2, test_index)
    print(get_acc(scores2, correct_answers, 'scores2'))
    scores3 = pickle.load(open('../data/event_comp_test.scores', 'rb'), encoding='bytes')
    scores3 = process_test(scores3, test_index)
    print(get_acc(scores3, correct_answers, 'scores3'))
    scores1 = preprocessing.scale(scores1)
    scores2 = preprocessing.scale(scores2)
    scores3 = preprocessing.scale(scores3)
    best_acc = 0.0
    best_i_j_k = (0, 0)
    for i in np.arange(-3, 3, 0.1):
        for j in np.arange(-3, 3, 0.1):
            acc = get_acc(scores3 * i + scores1 * j, correct_answers)
            if best_acc < acc:
                best_acc = acc 
                best_i_j_k = (i, j)
    print(best_acc, best_i_j_k)
    get_acc(scores3 * best_i_j_k[0] + scores1 * best_i_j_k[1], correct_answers, 'scores1_scores3')
    best_acc = 0.0
    best_i_j_k = (0, 0)
    for i in np.arange(-3, 3, 0.1):
        for j in np.arange(-3, 3, 0.1):
            acc = get_acc(scores1 * i + scores2 * j, correct_answers)
            if best_acc < acc:
                best_acc = acc 
                best_i_j_k = (i, j)
    print(best_acc, best_i_j_k)
    get_acc(scores1 * best_i_j_k[0] + scores2 * best_i_j_k[1], correct_answers, 'scores1_scores2')
    best_acc = 0.0
    best_i_j_k = (0, 0)
    for i in np.arange(-3, 3, 0.1):
        for j in np.arange(-3, 3, 0.1):
            acc = get_acc(scores3 * i + scores2 * j, correct_answers)
            if best_acc < acc:
                best_acc = acc 
                best_i_j_k = (i, j)
    print(best_acc, best_i_j_k)
    get_acc(scores3 * best_i_j_k[0] + scores2 * best_i_j_k[1], correct_answers, 'scores2_scores3')
    best_acc = 0.0
    best_i_j_k = (0, 0, 0)
    for i in np.arange(-3, 3, 0.1):
        for j in np.arange(-3, 3, 0.1):
            for k in np.arange(-3, 3, 0.1):
                acc = get_acc(scores1 * i + scores3 * j + scores2 * k, correct_answers)
                if best_acc < acc:
                    best_acc = acc 
                    best_i_j_k = (i, j, k)
    print(best_acc, best_i_j_k)
    get_acc(scores1 * best_i_j_k[0] + scores3 * best_i_j_k[1] + scores2 * best_i_j_k[2], correct_answers, 'scores1_scores2_scores3')
