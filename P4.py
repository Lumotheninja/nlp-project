from P1 import TrainProbabilities, START_TOK, STOP_TOK
from P2 import CRF
import numpy as np
from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b


class CRF(CRF):
    def __init__(self, training_path="data/EN/train", l2_param=0.1):
        self.training_path = training_path
        self.l2_param = l2_param
        super(CRF, self).__init__()

    def _forward(self, w, sentence):
        possible_y = self.train_probabilities.y_count.keys()

        # base case
        alpha_list = [{START_TOK: 0}]

        # START to all y
        last_y = START_TOK
        alpha_score = {}
        for next_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            # force constraint to never transition to start token
            # if next_y == START_TOK:
            #     w[transition_key] = -10000.
            # try:
            alpha_score[next_y] = w[transition_key]
            # except KeyError:
            #     alpha_score[next_y] = 0
        alpha_list.append(alpha_score)
        
        # all y to all y
        for index, x in enumerate(sentence[:-1]):
            alpha_score = {}
            for next_y in possible_y:
                alpha_score[next_y] = 0
                scores = []
                for last_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    emission_key = "emission:%s+%s"%(last_y, x)
                    # if next_y == START_TOK or last_y == STOP_TOK:
                    #     w[transition_key] = -10000.
                    # try:
                    score = w[transition_key] + alpha_list[index+1][last_y] + w[emission_key]
                    scores.append(score)
                    # except KeyError:
                    #     scores.append(-10000.)
                alpha_score[next_y] = log_sum_exp(np.array(scores))
            alpha_list.append(alpha_score)
        
        
        # all y to STOP
        next_y = STOP_TOK
        x = sentence[-1]
        alpha_score = {}
        alpha_score[next_y] = 0
        scores = []
        for last_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            emission_key = "emission:%s+%s"%(last_y, x)
            # if last_y == STOP_TOK:
            #     W[transition_key] = -10000.
            # try:
            score = w[transition_key] + alpha_list[-1][last_y] + w[emission_key]
            scores.append(score)
            # except KeyError:
            #     scores.append(-10000.)
        alpha_score[next_y] = log_sum_exp(np.array(scores))
        alpha_list.append(alpha_score)

        return alpha_list

    def calculate_loss(self, w, path):
        loss = 0
        
        running_x = []
        running_y = []
        with open(path,mode='r',encoding="utf-8") as file:
            for line in file:
                # End of a sequence
                if line=='\n':
                    forward_score = self._forward(w, running_x)
                    sentence = " ".join(running_x)
                    tag = " ".join(running_y)

                    # log the score stored at the last element of forward score
                    loss += forward_score[-1][STOP_TOK] - self._score(sentence, tag, w=w)
                    running_x = []
                    running_y = []
                else:
                    x,y = line.split()
                    running_x.append(x)
                    running_y.append(y)
        loss += self.l2_param * np.sum(np.power(list(w.values()), 2))
        return loss

    def _backward(self, w, sentence):
        
        possible_y = self.train_probabilities.y_count.keys()

        # base case
        beta_list = [{STOP_TOK: 0}]

        # all y to STOP
        next_y = STOP_TOK
        beta_score = {}
        x = sentence[-1]
        for last_y in possible_y:
            emission_key = "emission:%s+%s"%(last_y, x)
            transition_key = "transition:%s+%s"%(last_y,next_y)
            # if last_y == STOP_TOK:
            #     w[transition_key] = -10000.
            # try:
            beta_score[last_y] = w[transition_key] + w[emission_key]
            # except KeyError:
            #     beta_score[last_y] = 0
        beta_list.append(beta_score)
        
        # all y to all y
        for index, x in enumerate(sentence[:-1][::-1]):
            beta_score = {}
            for last_y in possible_y:
                beta_score[last_y] = 0
                scores = []
                for next_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    emission_key = "emission:%s+%s"%(last_y, x)
                    # if last_y == STOP_TOK or next_y == START_TOK:
                    #     w[transition_key] = -10000.
                    # try:
                    score = w[transition_key] + beta_list[index+1][next_y] + w[emission_key]
                    scores.append(score)
                    # except KeyError:
                    #     scores.append(-10000)
                beta_score[last_y] = log_sum_exp(np.array(scores))
            beta_list.append(beta_score)
        
        # START to all y
        last_y = START_TOK
        beta_score = {}
        sum_prob = []
        for next_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            # if next_y == START_TOK:
            #     w[transition_key] = -10000.
            # try:
            sum_prob.append(w[transition_key] + beta_list[-1][next_y])
            # except KeyError:
            #     sum_prob.append(-10000)
        beta_score[last_y] = log_sum_exp(np.array(sum_prob))
        beta_list.append(beta_score)

        return beta_list[::-1]
                
    def calculate_gradient(self, w, path):
        
        w_score = defaultdict(float)
        y_x_count = defaultdict(int)
        y0_y1_count = defaultdict(int)
        last_y = START_TOK
        running_x = []
        running_y = []

        with open(path,mode='r',encoding="utf-8") as file:
            for line in file:

                # End of a sequence
                if line=='\n':
                    # calculate forward backward scores and denom
                    y0_y1_count[(last_y,STOP_TOK)] += 1
                    forward_score = self._forward(w, running_x)
                    backward_score = self._backward(w, running_x)
                    # print(forward_score)
                    # print(backward_score)
                    
                    denom = np.exp(forward_score[-1][STOP_TOK])

                    # iterate through the y,x sequences in the sentence
                    for (y,x), counts in y_x_count.items():
                        emission_key = "emission:%s+%s"%(y, x)

                        '''
                        expected_counts = 0
                        # omit y as START and STOP
                        for index in range(1,len(forward_score)-1): 
                            # include all possible transitions
                            for next_y in forward_score[index+1].keys():
                                try:
                                    transition_key = "transition:%s+%s"%(y,next_y)
                                    expected_counts += forward_score[index][y] * backward_score[index+1][next_y] * np.exp(w[emission_key]) * np.exp(w[transition_key])
                                except:
                                    pass
                        w_score[emission_key] += expected_counts/denom - counts
                        '''
                        
                        expected_counts = 0
                        # omit y as START and STOP
                        for index in range(len(running_x)): 

                            # include all possible transitions
                            if x == running_x[index]:
                                try:
                                    expected_counts += np.exp(forward_score[index+1][y] + backward_score[index+1][y])
                                except KeyError as e:
                                    pass
                        w_score[emission_key] += expected_counts/denom - counts

                    # iterate through the y0,y1 sequences in the sentence
                    for (y0,y1), counts in y0_y1_count.items():
                        transition_key = "transition:%s+%s"%(y0,y1)

                        expected_counts = 0

                        # omit y_n as STOP
                        for index in range(0,len(forward_score)-1):

                            # START doesnt have emission
                            if index == 0:
                                try:
                                    expected_counts += np.exp(forward_score[index][y0] + backward_score[index+1][y1] + w[transition_key])
                                except KeyError:
                                    pass

                            # include y0 emission
                            else:
                                x = running_x[index-1]
                                emission_key = "emission:%s+%s"%(y0, x)
                                try:
                                    expected_counts += np.exp(forward_score[index][y0] + backward_score[index+1][y1] + w[emission_key] + w[transition_key])
                                except KeyError as e:
                                    pass

                        w_score[transition_key] += expected_counts/denom - counts
                    
                    # reset
                    y_x_count = defaultdict(int)
                    y0_y1_count = defaultdict(int)
                    last_y = START_TOK
                    running_x = []
                    running_y = []

                else:
                    x,y = line.split()
                    y_x_count[(y,x)] += 1
                    y0_y1_count[(last_y,y)] += 1
                    last_y = y
                    running_x.append(x)
                    running_y.append(y)
        w_score_temp = {k:v+2*self.l2_param*w[k] for k,v in w_score.items()}
        w_score.update(w_score_temp)
        return w_score
        
    def test_gradient(self, w, path, key, value =0.01):
        gradient = crf.calculate_gradient(w, path)[key]
        w_test = copy.deepcopy(w)
        w_test[key] += value
        
        diff = (crf.calculate_loss(w_test, path) - crf.calculate_loss(w, path))
        diff /= value
        return gradient, diff

    def callbackF(self, w):
        '''
        This function will be called by "fmin_l_bfgs_b"
        Arg:
        w: weights, numpy array
        '''
        loss = self.get_loss_grad(w)[0]
        print('Loss:{0:.4f}'.format(loss))

    def get_loss_grad(self, w):
        '''
        This function will be called by "fmin_l_bfgs_b"
        Arg:
        w: weights, numpy array
        Returns:
        loss: loss, float
        grads: gradients, numpy array
        '''
        w_dict = defaultdict(lambda:0)
        w_dict.update({k:v for k, v in zip(self.train_probabilities.f.keys(), w)})
        loss = self.calculate_loss(w_dict, self.training_path)
        print(loss)
        grads = self.calculate_gradient(w_dict, self.training_path)
        grads = np.array(list(grads.values()))
        print(grads)
        return loss, grads

    
    def train(self):
        # init_w = np.zeros(len(self.train_probabilities.f.keys()))
        # init_w = np.array(list(self.train_probabilities.f.values()))
        init_w = np.full(len(self.train_probabilities.f.keys()), 0)
        result = fmin_l_bfgs_b(self.get_loss_grad, init_w, pgtol=0.01, callback=self.callbackF)
        trained_weights = result[0]
        w_dict = defaultdict(lambda:-10000.)
        w_dict.update({k:v for k, v in zip(self.train_probabilities.f.keys(), trained_weights)})
        return w_dict
    
    def decode(self, w, test_path, out_path):
        self.train_probabilities.f = w
        self.apply_viterbi(test_path, out_path)

def log_sum_exp(vec):
    max_score = np.max(vec)
    return max_score + np.log(np.sum(np.exp(vec - max_score)))

if __name__ == "__main__":
    crf = CRF()
    trained_weights = crf.train()
    crf.decode(trained_weights, 'data/EN/dev.in', 'data/EN/dev.p4.out')
    