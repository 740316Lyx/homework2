# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Parameter, Module
# import math
# import torch.optim as optim

# class FNN(Module):
#     def __init__(self, hidden_size, dropout_p=0.2):
#         super(FNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.reset_parameters()

#     def forward(self, hidden):
#         hidden1 = torch.sigmoid(self.linear_one(hidden))
#         hidden2 = self.linear_two(hidden1)
#         return hidden2 + hidden

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

# class GNN(Module):
#     def __init__(self, hidden_size, T, dropout_p=0.2):
#         super(GNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.T = T
#         self.gate_size = 3 * hidden_size
#         self.w_ih = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.b_ih = Parameter(torch.Tensor(self.gate_size))
#         self.b_hh = Parameter(torch.Tensor(self.gate_size))
#         self.b_ah = Parameter(torch.Tensor(self.hidden_size))
#         self.w_ih_2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.w_hh_2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.b_ih_2 = Parameter(torch.Tensor(self.gate_size))
#         self.b_hh_2 = Parameter(torch.Tensor(self.gate_size))
#         self.b_ah_2 = Parameter(torch.Tensor(self.hidden_size))
#         self.reset_parameters()

#     def GNNCell(self, A, hidden, w_ih, w_hh, b_ih, b_hh, b_ah):
#         A_float = A.float()
#         input = torch.matmul(A_float.transpose(1, 2), hidden) + b_ah
#         gi = F.linear(input, w_ih, b_ih)
#         gh = F.linear(hidden, w_hh, b_hh)
#         i_r, i_i, i_n = gi.chunk(3, 2)
#         h_r, h_i, h_n = gh.chunk(3, 2)
#         resetgate = torch.sigmoid(i_r + h_r)
#         inputgate = torch.sigmoid(i_i + h_i)
#         newgate = torch.tanh(i_n + resetgate * h_n)
#         hy = newgate + inputgate * (hidden - newgate)
#         return hy

#     def forward(self, A, hidden):
#         hidden1 = self.GNNCell(A, hidden, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
#         hidden2 = self.GNNCell(A, hidden1, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
#         return hidden2

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

# class EventGraph_With_Args(Module):
#     def __init__(self, vocab_size, hidden_dim, word_vec, L2_penalty, MARGIN, LR, T, BATCH_SIZE=1000, dropout_p=0.2):
#         super(EventGraph_With_Args, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.batch_size = BATCH_SIZE
#         self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
#         self.embedding.weight.data = torch.from_numpy(word_vec).float()
#         self.gnn = GNN(self.hidden_dim, T)
#         self.linear_s_one = nn.Linear(hidden_dim, 1, bias=False)
#         self.linear_s_two = nn.Linear(hidden_dim, 1, bias=True)
#         self.linear_u_one = nn.Linear(hidden_dim, int(0.5 * hidden_dim), bias=True)
#         self.linear_u_one2 = nn.Linear(int(0.5 * hidden_dim), 1, bias=True)
#         self.linear_u_two = nn.Linear(hidden_dim, int(0.5 * hidden_dim), bias=True)
#         self.linear_u_two2 = nn.Linear(int(0.5 * hidden_dim), 1, bias=True)
#         self.multi = Parameter(torch.ones(3))
#         self.dropout = nn.Dropout(dropout_p)
#         self.loss_function = nn.MultiMarginLoss(margin=MARGIN)
#         model_grad_params = filter(lambda p: p.requires_grad, self.parameters())
#         train_params = list(map(id, self.embedding.parameters()))
#         tune_params = filter(lambda p: id(p) not in train_params, model_grad_params)
#         self.optimizer = optim.RMSprop([
#             {'params': tune_params},
#             {'params': self.embedding.parameters(), 'lr': LR * 0.06}
#         ], lr=LR, weight_decay=L2_penalty, momentum=0.2)

#     def compute_scores(self, hidden, metric='euclid'):
#         input_a = hidden[:, 0:8, :].repeat(1, 5, 1).view(5 * hidden.size(0), 8, -1)
#         input_b = hidden[:, 8:13, :]
#         u_a = F.relu(self.linear_u_one(input_a))
#         u_a2 = F.relu(self.linear_u_one2(u_a))
#         u_b = F.relu(self.linear_u_two(input_b))
#         u_b2 = F.relu(self.linear_u_two2(u_b))
#         u_c = torch.add(u_a2.view(5 * hidden.size(0), 8), u_b2.view(5 * hidden.size(0), 1))
#         weight = torch.exp(torch.tanh(u_c))
#         weight = (weight / torch.sum(weight, dim=1, keepdim=True)).view(-1, 8, 1)
#         weighted_input = torch.mul(input_a, weight)
#         a = torch.sum(weighted_input, dim=1)
#         b = input_b / 8.0
#         b = b.view(5 * hidden.size(0), -1)
#         if metric == 'dot':
#             scores = self.metric_dot(a, b)
#         elif metric == 'cosine':
#             scores = self.metric_cosine(a, b)
#         elif metric == 'euclid':
#             scores = self.metric_euclid(a, b)
#         elif metric == 'norm_euclid':
#             scores = self.metric_norm_euclid(a, b)
#         elif metric == 'manhattan':
#             scores = self.metric_manhattan(a, b)
#         elif metric == 'multi':
#             scores = (self.multi[0] * self.metric_euclid(a, b) + 
#                       self.multi[1] * self.metric_dot(a, b) + 
#                       self.multi[2] * self.metric_cosine(a, b))
#         return scores

#     def forward(self, input, A, metric='euclid', nn_type='gnn'):
#         input = input.long()  # 确保 input 是 Long 类型
#         hidden = self.embedding(input)
#         # hidden = self.embedding(input)[:, :13, :]
#         # 统一处理不同维度的输入 [修正点]
#         if hidden.dim() == 4:
#         # 验证阶段: [batch, num_nodes, num_events, emb_dim] -> 合并节点维度
#           hidden = hidden.reshape(hidden.size(0), -1, hidden.size(-1))
    
#     # 确保有足够的事件进行切片 [修正点]
#         if hidden.size(1) < 52:
#         # 不足52时填充0向量
#           pad_size = 52 - hidden.size(1)
#           padding = torch.zeros(hidden.size(0), pad_size, hidden.size(2)).to(hidden.device)
#           hidden = torch.cat([hidden, padding], dim=1)
#         else:
#         # 超过52时截断
#           hidden = hidden[:, :52, :]
#         print(hidden[:, 0:13, :].shape)
#         print(hidden[:, 13:26, :].shape)
#         print(1)
#         print(hidden[:, 26:39, :].shape)
#         print(hidden[:, 39:52, :].shape) 
#         hidden = torch.cat((hidden[:, 0:13, :], hidden[:, 13:26, :], hidden[:, 26:39, :], hidden[:, 39:52, :]), dim=2)  
#         # 原拼接行删除
#         hidden = hidden[:, :13, :]    # 只取前 13 个事件
#         print(hidden.shape) 
#         if nn_type == 'gnn':
#             hidden = self.gnn(A, hidden)
#         scores = self.compute_scores(hidden, metric)
#         return scores

#     def predict(self, input, A, targets, dev_index, metric='euclid'):
#         scores = self.forward(input, A, metric)
#         scores[dev_index] = -100.0
#         sorted_scores, L = torch.sort(scores, descending=True)
#         num_correct0 = (L[:, 0] == targets).sum().item()
#         num_correct1 = (L[:, 1] == targets).sum().item()
#         num_correct2 = (L[:, 2] == targets).sum().item()
#         num_correct3 = (L[:, 3] == targets).sum().item()
#         num_correct4 = (L[:, 4] == targets).sum().item()
#         samples = len(targets)
#         accuracy0 = num_correct0 / samples * 100.0 
#         accuracy1 = num_correct1 / samples * 100.0 
#         accuracy2 = num_correct2 / samples * 100.0 
#         accuracy3 = num_correct3 / samples * 100.0 
#         accuracy4 = num_correct4 / samples * 100.0 
#         return accuracy0, accuracy1, accuracy2, accuracy3, accuracy4

#     def metric_dot(self, v0, v1):
#         return torch.sum(v0 * v1, dim=1).view(-1, 5)

#     def metric_cosine(self, v0, v1):
#         return F.cosine_similarity(v0, v1).view(-1, 5)

#     def metric_euclid(self, v0, v1):
#         return -torch.norm(v0 - v1, p=2, dim=1).view(-1, 5)

#     def metric_norm_euclid(self, v0, v1):
#         v0_norm = v0 / torch.norm(v0, p=2, dim=1, keepdim=True)
#         v1_norm = v1 / torch.norm(v1, p=2, dim=1, keepdim=True)
#         return -torch.norm(v0_norm - v1_norm, p=2, dim=1).view(-1, 5)

#     def metric_manhattan(self, v0, v1):
#         return -torch.sum(torch.abs(v0 - v1), dim=1).view(-1, 5)

#     def correct_answer_position(self, L, correct_answers):
#         num_correct1 = (L[:, 0] == correct_answers).sum().item()
#         num_correct2 = (L[:, 1] == correct_answers).sum().item()
#         num_correct3 = (L[:, 2] == correct_answers).sum().item()
#         num_correct4 = (L[:, 3] == correct_answers).sum().item()
#         num_correct5 = (L[:, 4] == correct_answers).sum().item()
#         print(f"{num_correct1} / {len(correct_answers)} 1st max correct: {num_correct1 / len(correct_answers) * 100.0}")
#         print(f"{num_correct2} / {len(correct_answers)} 2nd max correct: {num_correct2 / len(correct_answers) * 100.0}")
#         print(f"{num_correct3} / {len(correct_answers)} 3rd max correct: {num_correct3 / len(correct_answers) * 100.0}")
#         print(f"{num_correct4} / {len(correct_answers)} 4th max correct: {num_correct4 / len(correct_answers) * 100.0}")
#         print(f"{num_correct5} / {len(correct_answers)} 5th max correct: {num_correct5 / len(correct_answers) * 100.0}")

#     def predict_with_minibatch(self, input, A, targets, dev_index, metric='euclid'):
#         scores = torch.zeros(len(targets), 5).to(input.device)
#         for i in range(0, len(targets), self.batch_size):
#             end = min(i + self.batch_size, len(targets))
#             batch_input = input[i:end]
#             batch_A = A[i:end]
#             batch_scores = self.forward(batch_input, batch_A, metric)
#             scores[i:end] = batch_scores
#         scores[dev_index] = -100.0
#         sorted_scores, L = torch.sort(scores, descending=True)
#         accuracy0 = (L[:, 0] == targets).sum().item() / len(targets) * 100.0 
#         accuracy1 = (L[:, 1] == targets).sum().item() / len(targets) * 100.0 
#         accuracy2 = (L[:, 2] == targets).sum().item() / len(targets) * 100.0 
#         accuracy3 = (L[:, 3] == targets).sum().item() / len(targets) * 100.0 
#         accuracy4 = (L[:, 4] == targets).sum().item() / len(targets) * 100.0 
#         return accuracy0, accuracy1, accuracy2, accuracy3, accuracy4

#     def weights_init(self, m):
#         if isinstance(m, nn.GRU):
#             nn.init.xavier_uniform_(m.weight_hh_l0)
#             nn.init.xavier_uniform_(m.weight_ih_l0)
#             nn.init.constant_(m.bias_hh_l0, 0)
#             nn.init.constant_(m.bias_ih_l0, 0)
#         elif isinstance(m, GNN):
#             nn.init.xavier_uniform_(m.w_hh)
#             nn.init.xavier_uniform_(m.w_ih)
#             nn.init.xavier_uniform_(m.w_hh_2)
#             nn.init.xavier_uniform_(m.w_ih_2)
#             nn.init.constant_(m.b_hh, 0)
#             nn.init.constant_(m.b_ih, 0)
#             nn.init.constant_(m.b_ah, 0)
#             nn.init.constant_(m.b_hh_2, 0)
#             nn.init.constant_(m.b_ih_2, 0)
#             nn.init.constant_(m.b_ah_2, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)