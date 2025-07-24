import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Module
import math
from utils import trans_to_cuda
import torch.optim as optim
class EventChain(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, word_vec, num_layers=1, bidirectional=False):
        super(EventChain, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 1 if not self.bidirectional else 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data = torch.from_numpy(word_vec).float()
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=0.2, bidirectional=self.bidirectional)
        self.linear_s_one = nn.Linear(hidden_dim * self.num_directions, 1, bias=False)
        self.linear_s_two = nn.Linear(hidden_dim * self.num_directions, 1, bias=True)
        self.linear_u_one = nn.Linear(hidden_dim * self.num_directions, 1, bias=False)
        self.linear_u_two = nn.Linear(hidden_dim * self.num_directions, 1, bias=True)
        self.loss_function = nn.MultiMarginLoss(margin=0.015)
        model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
        self.optimizer = optim.RMSprop([
            {'params': tune_params},
            {'params': self.embedding.parameters(), 'lr': 0.0001 * 0.06}
        ], lr=0.0001, weight_decay=1e-8, momentum=0.2)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 60], gamma=0.5)

    def compute_scores(self, output):
        output = output.transpose(0, 1)
        a = self.linear_s_one(output[:, 0:8, :])
        b = self.linear_s_two(output[:, 8, :])
        c = torch.add(a.view(-1, 8), b)
        scores = torch.sigmoid(c)
        u_a = self.linear_u_one(output[:, 0:8, :])
        u_b = self.linear_u_two(output[:, 8, :])
        u_c = torch.add(u_a.view(-1, 8), u_b)
        weight = torch.exp(torch.tanh(u_c))
        weight = weight / torch.sum(weight, 1, keepdim=True)
        scores = torch.sum(torch.mul(scores, weight), 1).view(-1, 5)
        return scores

    def forward(self, input):
        hidden = self.embedding(input.long())
        hidden = torch.cat((hidden[:, 0:13, :], hidden[:, 13:26, :], hidden[:, 26:39, :], hidden[:, 39:52, :]), 2)
        input_a = hidden[:, 0:8, :].repeat(1, 5, 1).view(5 * len(hidden), 8, -1)
        input_b = hidden[:, 8:13, :].contiguous().view(-1, 1, 512)
        hidden = torch.cat((input_a, input_b), 1)
        self.hidden = self.init_hidden(len(hidden))
        output = hidden.transpose(0, 1)
        output, self.hidden = self.gru(output, self.hidden)
        scores = self.compute_scores(output)
        return scores

    def predict(self, input, targets):
        scores = self.forward(input)
        sorted_scores, L = torch.sort(scores, descending=True)
        num_correct0 = (L[:, 0] == targets).sum().item()
        num_correct1 = (L[:, 1] == targets).sum().item()
        num_correct2 = (L[:, 2] == targets).sum().item()
        num_correct3 = (L[:, 3] == targets).sum().item()
        num_correct4 = (L[:, 4] == targets).sum().item()
        samples = len(targets)
        accuracy0 = num_correct0 / samples * 100.0 
        accuracy1 = num_correct1 / samples * 100.0 
        accuracy2 = num_correct2 / samples * 100.0 
        accuracy3 = num_correct3 / samples * 100.0 
        accuracy4 = num_correct4 / samples * 100.0 
        return accuracy0, accuracy1, accuracy2, accuracy3, accuracy4

    def predict_with_minibatch(self, input, targets):
        scores = torch.zeros(len(targets), 5).cuda()
        for i in range(0, len(targets), 1000):
            end = min(i + 1000, len(targets))
            scores_temp = self.forward(input[i:end])
            scores[i:end] = scores_temp
        sorted_scores, L = torch.sort(scores, descending=True)
        num_correct0 = (L[:, 0] == targets).sum().item()
        num_correct1 = (L[:, 1] == targets).sum().item()
        num_correct2 = (L[:, 2] == targets).sum().item()
        num_correct3 = (L[:, 3] == targets).sum().item()
        num_correct4 = (L[:, 4] == targets).sum().item()
        samples = len(targets)
        accuracy0 = num_correct0 / samples * 100.0 
        accuracy1 = num_correct1 / samples * 100.0 
        accuracy2 = num_correct2 / samples * 100.0 
        accuracy3 = num_correct3 / samples * 100.0 
        accuracy4 = num_correct4 / samples * 100.0 
        return accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, scores

    def init_hidden(self, size):
        hidden = torch.zeros(self.num_layers * self.num_directions, size, self.hidden_dim)
        return trans_to_cuda(hidden)

    def weights_init(self, m):
        if isinstance(m, nn.GRU):
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.constant_(m.bias_hh_l0, 0)
            nn.init.constant_(m.bias_ih_l0, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

class Word2VecAttention(nn.Module):
    def __init__(self):
        super(Word2VecAttention, self).__init__()
        self.linear_u_one = nn.Linear(512, 1, bias=False)
        self.linear_u_one2 = nn.Linear(512, 1, bias=False)
        self.linear_u_two = nn.Linear(512, 1, bias=True)
        self.linear_u_two2 = nn.Linear(512, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def compute_scores(self, input_data):
        weight = torch.zeros((len(input_data), 8, 1)).fill_(1./8).to(input_data.device)
        weighted_input = torch.mul(input_data[:, 0:8, :], weight)
        a = torch.sum(weighted_input, 1)
        b = input_data[:, 8, :] / 8.0
        scores = -torch.norm(a - b, 2, 1).view(-1, 5)
        return scores

    def forward(self, input_data):
        return self.compute_scores(input_data)

    def correct_answer_position(self, L, correct_answers):
        num_correct1 = (L[:, 0] == correct_answers).sum().item()
        num_correct2 = (L[:, 1] == correct_answers).sum().item()
        num_correct3 = (L[:, 2] == correct_answers).sum().item()
        num_correct4 = (L[:, 3] == correct_answers).sum().item()
        num_correct5 = (L[:, 4] == correct_answers).sum().item()
        print(f"{num_correct1} / {len(correct_answers)} 1st max correct: {num_correct1 / len(correct_answers) * 100.0}")
        print(f"{num_correct2} / {len(correct_answers)} 2nd max correct: {num_correct2 / len(correct_answers) * 100.0}")
        print(f"{num_correct3} / {len(correct_answers)} 3rd max correct: {num_correct3 / len(correct_answers) * 100.0}")
        print(f"{num_correct4} / {len(correct_answers)} 4th max correct: {num_correct4 / len(correct_answers) * 100.0}")
        print(f"{num_correct5} / {len(correct_answers)} 5th max correct: {num_correct5 / len(correct_answers) * 100.0}")

    def predict(self, input_data, targets):
        scores = self.forward(input_data)
        sorted_scores, L = torch.sort(scores, descending=True)
        self.correct_answer_position(L, targets)
        selections = L[:, 0]
        pickle.dump((selections != targets), open('../data/test.answer', 'wb'))
        num_correct = (selections == targets).sum().item()
        accuracy = num_correct / len(targets) * 100.0
        return accuracy

    def weights_init(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.GRU):
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.constant_(m.bias_hh_l0, 0)
            nn.init.constant_(m.bias_ih_l0, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
