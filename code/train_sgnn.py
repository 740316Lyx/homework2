import torch
import pickle
import time
from utils import Data_data, trans_to_cuda, get_hash_for_word
from gnn_with_args import EventGraph_With_Args
import sys

def train(dev_index, word_vec, ans, train_data, dev_data, test_data, L2_penalty, MARGIN, LR, T, BATCH_SIZE, EPOCHES, PATIENTS, HIDDEN_DIM, METRIC='euclid'):
    model = EventGraph_With_Args(
        vocab_size=len(word_vec),
        hidden_dim=HIDDEN_DIM,
        word_vec=word_vec,
        L2_penalty=L2_penalty,
        MARGIN=MARGIN,
        LR=LR,
        T=T,
        BATCH_SIZE=BATCH_SIZE
    )
    model = trans_to_cuda(model)
    model.optimizer.zero_grad()
    acc_list = []
    best_acc = 0.0
    best_epoch = 0
    print('start training')
    EPO = 0
    start = time.time()
    while True:
        patient = 0
        for epoch in range(EPOCHES):
            data, epoch_flag = train_data.next_batch(BATCH_SIZE)
            model.train()
            scores = model(data[1], data[0], metric=METRIC)
            loss = model.loss_function(scores, data[2])
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            dev_input, dev_A, dev_targets = dev_data.all_data()
            dev_input = trans_to_cuda(dev_input)
            dev_A = trans_to_cuda(dev_A)
            dev_targets = trans_to_cuda(dev_targets)
            model.eval()
            with torch.no_grad():
                accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = model.predict(dev_input, dev_A, dev_targets, dev_index, metric=METRIC)
            if (EPOCHES * EPO + epoch) % 50 == 0:
                print(f'Epoch {EPOCHES * EPO + epoch} : Eval Acc: {accuracy:.4f}, {accuracy1:.4f}, {accuracy2:.4f}, {accuracy3:.4f}, {accuracy4:.4f}, {METRIC}')
            acc_list.append((time.time() - start, accuracy))
            if best_acc < accuracy:
                best_acc = accuracy
                if best_acc >= 52.7:
                    torch.save(model.state_dict(), f'../data/gnn_{METRIC}_acc_{best_acc}_.model')
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
    return best_acc, best_epoch

def main():
    dev_data = Data_data(pickle.load(open('../data/corpus_index_dev_with_args_all_chain.data', 'rb')))
    test_data = Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data', 'rb')))
    train_data = Data_data(pickle.load(open('../data/corpus_index_train0_with_args_all_chain.data', 'rb')))
    ans = pickle.load(open('../data/dev.answer', 'rb'))
    dev_index = pickle.load(open('../data/dev_index.pickle', 'rb'))
    print('train data prepare done')
    word_id, id_vec, word_vec = get_hash_for_word('../data/deepwalk_128_unweighted_with_args.txt', '../data/encoding_with_args.csv')
    print('word vector prepare done')
    if len(sys.argv) == 9:
        L2_penalty, MARGIN, LR, T, BATCH_SIZE, EPOCHES, PATIENTS, METRIC = sys.argv[1:]
        L2_penalty = float(L2_penalty)
        MARGIN = float(MARGIN)
        LR = float(LR)
        T = int(T)
        BATCH_SIZE = int(BATCH_SIZE)
        EPOCHES = int(EPOCHES)
        PATIENTS = int(PATIENTS)
    else:
        HIDDEN_DIM = 128 * 4
        L2_penalty = 0.00001
        LR = 0.0001
        T = 2
        MARGIN = 0.015
        BATCH_SIZE = 1000
        EPOCHES = 520
        PATIENTS = 500
        METRIC = 'euclid'
        if METRIC == 'euclid':
            L2_penalty = 0.00001
            LR = 0.0001
            BATCH_SIZE = 1000
            MARGIN = 0.015
            PATIENTS = 500
        elif METRIC == 'dot':
            MARGIN = 0.5
        elif METRIC == 'cosine':
            MARGIN = 0.05
        elif METRIC == 'norm_euclid':
            MARGIN = 0.07
        elif METRIC == 'manhattan':
            MARGIN = 4.5
        elif METRIC == 'multi':
            MARGIN = 0.015
        elif METRIC == 'nonlinear':
            MARGIN = 0.015
    start_time = time.time()
    best_acc, best_epoch = train(dev_index, word_vec, ans, train_data, dev_data, test_data, L2_penalty, MARGIN, LR, T, BATCH_SIZE, EPOCHES, PATIENTS, HIDDEN_DIM, METRIC)
    end_time = time.time()
    print(f"Run time: {end_time - start_time} s")
    with open('best_result.txt', 'a') as f:
        f.write(f'Best Acc: {best_acc}, Epoch {best_epoch}, L2_penalty={L2_penalty}, MARGIN={MARGIN}, LR={LR}, T={T}, BATCH_SIZE={BATCH_SIZE}, EPOCHES={EPOCHES}, PATIENTS={PATIENTS}, HIDDEN_DIM={HIDDEN_DIM}, METRIC={METRIC}\n')

if __name__ == '__main__':
    main()
