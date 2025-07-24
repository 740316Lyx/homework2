import torch
import pickle
import time
from utils import Data_data, trans_to_cuda, get_hash_for_word
from event_chain import EventChain
from event_chain import Word2VecAttention
from sklearn import preprocessing

def train_event_chain():
    model = EventChain(
        embedding_dim=512,
        hidden_dim=512,
        vocab_size=len(word_vec),
        word_vec=word_vec,
        num_layers=1,
        bidirectional=False
    )
    model = trans_to_cuda(model)
    acc_list = []
    best_acc = 0.0
    best_epoch = 0
    print('start training')
    EPO = 0
    start = time.time()
    while True:
        patient = 0
        for epoch in range(EPOCHES):
            model.optimizer.zero_grad()
            data, epoch_flag = train_data.next_batch(BATCH_SIZE)
            scores = model(data[1])
            loss = model.loss_function(scores, data[2])
            loss.backward()
            model.optimizer.step()
            dev_input, dev_A, dev_targets = dev_data.all_data()
            accuracy, accuracy1, accuracy2, accuracy3, accuracy4, scores2 = model.predict_with_minibatch(dev_input, dev_targets)
            print(f'Epoch {EPOCHES * EPO + epoch} : Eval Acc: {accuracy}, {accuracy1}, {accuracy2}, {accuracy3}, {accuracy4}')
            acc_list.append((time.time() - start, accuracy))
            if best_acc < accuracy:
                best_acc = accuracy
                if best_acc >= 51:
                    torch.save(model.state_dict(), f'../data/event_chain_acc_{best_acc}_.model')
                best_epoch = EPOCHES * EPO + epoch + 1
                patient = 0
            else:
                patient += 1
            if patient > PATIENTS:
                break
        if epoch == (EPOCHES - 1):
            EPO += 1
            continue
        else:
            break
    print(f'Epoch {best_epoch} : Best Acc: {best_acc}')
    pickle.dump(acc_list, open('../data/lstm_acc_list.pickle', 'wb'), 2)
    return best_acc, best_epoch

def main():
    model = EventChain(
        embedding_dim=512,
        hidden_dim=512,
        vocab_size=len(word_vec),
        word_vec=word_vec,
        num_layers=1,
        bidirectional=False
    )
    model.load_state_dict(torch.load('../data/event_chain_acc_50.98999786376953_.model'))
    test_data = Data_data(pickle.load(open('../data/corpus_index_test_with_args_all.data', 'rb')))
    data = test_data.all_data()
    correct_answers = data[2].cpu().numpy()
    scores1 = model(data[1]).cpu().numpy()
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

if __name__ == '__main__':
    HIDDEN_DIM = 512
    L2_penalty = 1e-8
    LR = 0.0001
    MARGIN = 0.015
    BATCH_SIZE = 1000
    EPOCHES = 520
    PATIENTS = 500
    DROPOUT = 0.2
    dev_data = Data_data(pickle.load(open('../data/corpus_index_dev_with_args_all_chain.data', 'rb')))
    test_data = Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data', 'rb')))
    train_data = Data_data(pickle.load(open('../data/corpus_index_train0_with_args_all_chain.data', 'rb')))
    print('train data prepare done')
    word_id, id_vec, word_vec = get_hash_for_word('../data/deepwalk_128_unweighted_with_args.txt', '../data/encoding_with_args.csv')
    print('word vector prepare done')
    best_acc, best_epoch = train_event_chain()
    print(f'Epoch {best_epoch} : Best Acc: {best_acc}')
    pickle.dump(acc_list, open('../data/lstm_acc_list.pickle', 'wb'), 2)
