import os
import numba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_potentials(transcript):
    start_label, K = transcript[0], len(transcript)
    start_prob = torch.zeros(K)
    start_prob[start_label] = 1.0
    log_start = torch.log(start_prob)
    #log_prior = torch.log(prior_prob + 1e-7)
    
    trans_prob = torch.zeros(K, K)
    for i in range(len(transcript) - 1):
        curr_label, next_label = transcript[i], transcript[i + 1]
        trans_prob[curr_label, curr_label] = 0.5
        trans_prob[curr_label, next_label] = 0.5
    end_label = transcript[-1]
    trans_prob[end_label, end_label] = 0.5
    
    log_trans = torch.log(trans_prob)
    #log_trans = torch.log(trans_prob + 1e-7)
    
    end_prob = torch.zeros(K)
    end_prob[end_label] = 1.0
    log_end = torch.log(end_prob)
    #log_end = torch.log(end_prob + 1e-7)
    return log_start, log_trans, log_end

def viterbi(log_probs, log_start, log_trans, log_end):
    
    """
    log_probs: T x K, log_trans: K x K
    """
    T = log_probs.shape[0]
    score = log_start + log_probs[0, :]
    traceback, path = [], []
    for t in range(1, T):
        alpha = score[:, None] + log_trans + log_probs[t]
        score, inds = torch.max(alpha, 0)
        traceback.append(inds)
    score += log_end
    traceback = torch.stack(traceback)
    
    path.append(score.argmax().item())
    for t in range(traceback.shape[0] - 1, -1, -1):
        curr_state = path[-1]
        prev_state = traceback[t][curr_state].item()
        #print('Row {} frame {} --> {} curr_state {} --> prev_state {}'.format(t, t+1, t, curr_state, prev_state))
        path.append(prev_state)

    path = list(reversed(path))
    return np.array(path), score.max().item()

def viterbi_pytorch(log_prior, log_trans, log_end, emissions, mask, diff = False):
    
    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert mask[0].all()
    
    seq_length, batch_size = mask.shape
    score = log_prior + emissions[0] #batch_size x K
    history = []
    
    if diff == True:        
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)                            #batch_size x K x 1
            broadcast_emission = emissions[i].unsqueeze(1)                  #batch_size x 1 x K
            next_score = broadcast_score + log_trans + broadcast_emission #batch_size x K x K

            indices = next_score.argmax(dim=1) - next_score.max(dim=1)[0].detach() + next_score.max(dim=1)[0]
            next_score = next_score.max(dim=1)[0]
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            best_last_tag = score[idx].argmax(0) - score[idx].max().detach() + score[idx].max()
            best_tags = [best_last_tag]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][int(best_tags[-1])]
                best_tags.append(best_last_tag)

            best_tags.reverse()
            best_tags_list.append(torch.stack(best_tags))
            
    else:
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + log_trans + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
           
        score += log_end
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
                
            best_tags.reverse()
            best_tags_list.append(best_tags)
            
    return best_tags_list, score.max().item()

@numba.jit(nopython=True)
def cumsum(arr):
    output = np.zeros_like(arr)
    for ind in range(output.shape[0]):
        output[ind] = np.cumsum(arr[ind])
    return output

@numba.jit(nopython=True)
def segmental_viterbi(log_probs, log_start, log_trans, log_end, length_matrix):
    
    L = length_matrix.shape[1]
    K, T = log_probs.shape

    delta = np.zeros((K, T))
    psi = np.zeros((K, T), dtype = np.int64)
    tau = np.zeros((K, T), dtype = np.int64)
    path = np.zeros(T, dtype = np.int64)
    
    delta[:, 0] = log_start + log_probs[:, 0]
    psi[:, 0] = 0 #the best predecessor state, given that we ended up in state j at time t
    tau[:, 0] = 0 #the best duration of current state
    
    for t in range(1, T):
        
        t0 = max(0, t - L + 1)
        log_probs_ = log_probs[:, np.arange(t, t0, -1)] #numba doesn't recognize log_probs[:, arange(t, t0, -1)]     
        log_probs_ = cumsum(log_probs_) #equivalent to np.cumsum(log_probs_, axis=1)
        for j in range(K):

            Lm = min(t, L - 1)
            delta_, psi_ = np.zeros(Lm), np.zeros(Lm)
            for d in range(0, Lm):
                score = delta[:, t - d - 1] + log_trans[:, j]
                delta_[d] = np.max(score)
                psi_[d] = np.argmax(score)
                #print('time {} state {} d {} score {} delta_ {} psi_ {}'.format(t, j, d, score, delta_, psi_))

                #length indexes from 1, but python indexes from 0
                delta_[d] = delta_[d] + length_matrix[j, d] + log_probs_[j, d]
                #print('===After duration && observation, delta_ {}'.format(delta_))

            delta[j, t] = np.max(delta_)
            tau[j, t] = np.argmax(delta_) #The optimal duration for the current state
            psi[j, t] = psi_[tau[j, t]]   #The best previous state at time t
            
#         print(delta[:, 0:t+1])
#         print(tau[:, 0:t+1])
#         print(psi[:, 0:t+1])
#         print()

    if log_end != None:
        delta[:, T-1] += log_end

    path[T - 1] = delta[:, T - 1].argmax()
    durations = np.zeros(T, dtype = np.int64)
    durations[T - 1] = tau[path[T - 1], T - 1]
    count = 0
    for t in range(T - 2, 0, -1):
        #print('Curr time %d, its ancestor duration is %d'%(t, durations[t + 1]))
        if durations[t + 1] > 0:
            #print('Action remain same as its ancestor state %d' %path[t+1])
            path[t] = path[t + 1]
            durations[t] = durations[t + 1] - 1
            count += 1
        else:
            #print('--Action transition happens--')
            path[t] = psi[path[t + count + 1], t + count + 1]
            durations[t] = tau[path[t], t]
            count = 0
        #print()

    path[0] = delta[:, 0].argmax()
    return path, delta[:, -1].max()

#PyTorch implementation
# start_transitions = torch.randn(5)
# transitions = torch.randn(5, 5)
# emissions = torch.randn(7, 2, 5)
# mask = torch.ByteTensor(7, 2)

# a = torch.ByteTensor([1, 1, 1, 1, 1, 1, 1])[:, None]
# b = torch.ByteTensor([1, 1, 1, 0, 0, 0, 0])[:, None]
# mask = torch.cat((a, b), 1)

# traceback = []
# score = start_transitions + emissions[0]
# for i in range(1, emissions.shape[0]):
#     broadcast_score = score.unsqueeze(2)
#     broadcast_emissions = emissions[i].unsqueeze(1)
#     next_score = broadcast_score + transitions + broadcast_emissions
    
#     next_score, indices = next_score.max(1)
#     print(i, mask[i])
#     print(score)
#     print(next_score)
#     score = torch.where(mask[i].unsqueeze(-1), next_score, score)
#     print(score)
#     print('\n')
#     traceback.append(indices)

# seq_ends = mask.long().sum(0) - 1
# best_tags_list = []
# for idx in range(emissions.shape[1]):
    
#     best_last_tag = score[idx].argmax()
#     best_tags = [best_last_tag.item()]
    
#     for trace in reversed(traceback[:seq_ends[idx]]):
#         tmp = trace[idx][best_tags[-1]].item()
#         print('traceback {}'.format(trace[idx]))
#         best_tags.append(tmp)
#         print('best_tags {}'.format(best_tags))
        
#     best_tags.reverse()
#     best_tags_list.append(best_tags)
    
# print(best_tags_list)

# def batch_viterbi(start_transitions, transitions, emissions, mask, differentiable=True):
    
#     assert emissions.dim() == 3 and mask.dim() == 2
#     assert emissions.shape[:2] == mask.shape
#     assert mask[0].all()

#     seq_length, batch_size = mask.shape
#     score = start_transitions + emissions[0] #batch_size x K
#     history = []
    
#     if differentiable == True:        
#         for i in range(1, seq_length):
#             broadcast_score = score.unsqueeze(2)                            #batch_size x K x 1
#             broadcast_emission = emissions[i].unsqueeze(1)                  #batch_size x 1 x K
#             next_score = broadcast_score + transitions + broadcast_emission #batch_size x K x K

#             indices = next_score.argmax(dim=1) - next_score.max(dim=1)[0].detach() + next_score.max(dim=1)[0]
#             next_score = next_score.max(dim=1)[0]
#             score = torch.where(mask[i].unsqueeze(1), next_score, score)
#             history.append(indices)

#         seq_ends = mask.long().sum(dim=0) - 1
#         best_tags_list = []

#         for idx in range(batch_size):
#             best_last_tag = score[idx].argmax(0) - score[idx].max().detach() + score[idx].max()
#             best_tags = [best_last_tag]

#             for hist in reversed(history[:seq_ends[idx]]):
#                 best_last_tag = hist[idx][int(best_tags[-1])]
#                 best_tags.append(best_last_tag)

#             best_tags.reverse()
#             best_tags_list.append(torch.stack(best_tags))
            
#     else:
#         for i in range(1, seq_length):
#             broadcast_score = score.unsqueeze(2)
#             broadcast_emission = emissions[i].unsqueeze(1)
#             next_score = broadcast_score + transitions + broadcast_emission
#             next_score, indices = next_score.max(dim=1)
            
#             score = torch.where(mask[i].unsqueeze(1), next_score, score)
#             history.append(indices)
            
#         seq_ends = mask.long().sum(dim=0) - 1
#         best_tags_list = []

#         for idx in range(batch_size):
#             _, best_last_tag = score[idx].max(dim=0)
#             best_tags = [best_last_tag.item()]
            
#             for hist in reversed(history[:seq_ends[idx]]):
#                 best_last_tag = hist[idx][best_tags[-1]]
#                 best_tags.append(best_last_tag.item())
                
#             best_tags.reverse()
#             best_tags_list.append(best_tags)
            
#     return best_tags_list

# emissions.requires_grad_()
# batch_viterbi(start_transitions, transitions, emissions, mask, differentiable=True)

# batch_viterbi(start_transitions, transitions, emissions, mask, differentiable=False)

# import numpy as np

# def viterbi_decoding(transitions, emissions):
#     traceback_table = np.zeros_like(emissions, dtype=int)
#     scores = emissions[0]
#     for t in range(1, emissions.shape[0]):
#         score_with_transition = scores[:, None] + transitions
#         scores = score_with_transition.max(0) + emissions[t]
#         traceback_table[t] = np.argmax(score_with_transition, 0)
        
#     optimal_path = [scores.argmax()]
#     for t in range(traceback_table.shape[0] - 1, 0, -1):
#         optimal_path.append(traceback_table[t][optimal_path[-1]])

#     optimal_path = list(reversed(optimal_path))
#     return optimal_path, scores.max()

# seq_length = 20
# num_states = 7
# emissions = np.random.randn(seq_length, num_states)
# transitions = np.random.randn(num_states, num_states)

# optimal_path, optimal_score = viterbi_decoding(transitions, emissions)
# print(optimal_path, optimal_score)