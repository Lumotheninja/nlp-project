from P1 import TrainProbabilities, START_TOK, STOP_TOK
from P2 import CRF
import numpy as np
from collections import defaultdict
import copy


class CRF(CRF):

    def _forward(self, w, sentence):
        possible_y = self.train_probabilities.y_count.keys()

        # base case
        alpha_list = [{START_TOK: 1}]

        # START to all y
        last_y = START_TOK
        alpha_score = {}
        for next_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                alpha_score[next_y] = np.exp(w[transition_key])
            except KeyError:
                alpha_score[next_y] = 0
        alpha_list.append(alpha_score)
        
        # all y to all y
        for index, x in enumerate(sentence[:-1]):
            alpha_score = {}
            for next_y in possible_y:
                alpha_score[next_y] = 0
                for last_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    emission_key = "emission:%s+%s"%(last_y, x)
                    try:
                        alpha_score[next_y] += np.exp(w[transition_key]) * alpha_list[index+1][last_y] * np.exp(w[emission_key])
                    except KeyError:
                        pass
            alpha_list.append(alpha_score)
        
        
        # all y to STOP
        next_y = STOP_TOK
        x = sentence[-1]
        alpha_score = {}
        alpha_score[next_y] = 0
        for last_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            emission_key = "emission:%s+%s"%(last_y, x)
            try:
                alpha_score[next_y] += np.exp(w[transition_key]) * alpha_list[-1][last_y] * np.exp(w[emission_key])
            except KeyError:
                pass
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
                    loss += np.log(forward_score[-1][STOP_TOK]) - self._score(sentence, tag, w=w)
                    running_x = []
                    running_y = []
                else:
                    x,y = line.split()
                    running_x.append(x)
                    running_y.append(y)
        return loss

    def _backward(self, w, sentence):
        
        possible_y = self.train_probabilities.y_count.keys()

        # base case
        beta_list = [{STOP_TOK: 1}]

        # all y to STOP
        next_y = STOP_TOK
        beta_score = {}
        x = sentence[-1]
        for last_y in possible_y:
            emission_key = "emission:%s+%s"%(last_y, x)
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                beta_score[last_y] = np.exp(w[transition_key]) * np.exp(w[emission_key])
            except KeyError:
                beta_score[last_y] = 0
        beta_list.append(beta_score)
        
        # all y to all y
        for index, x in enumerate(sentence[:-1][::-1]):
            beta_score = {}
            for last_y in possible_y:
                beta_score[last_y] = 0
                for next_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    emission_key = "emission:%s+%s"%(last_y, x)
                    try:
                        beta_score[last_y] += np.exp(w[transition_key]) * beta_list[index+1][next_y] * np.exp(w[emission_key])
                    except KeyError:
                        pass
            beta_list.append(beta_score)
        
        # START to all y
        last_y = START_TOK
        beta_score = {}
        sum_prob = 0
        for next_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                sum_prob += np.exp(w[transition_key]) * beta_list[-1][next_y]
            except KeyError:
                pass
        beta_score[last_y] = sum_prob
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
                    
                    denom = forward_score[-1][STOP_TOK]

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
                                    expected_counts += forward_score[index+1][y] * backward_score[index+1][y]
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
                                    expected_counts += forward_score[index][y0] * backward_score[index+1][y1]* np.exp(w[transition_key])
                                except KeyError:
                                    pass

                            # include y0 emission
                            else:
                                x = running_x[index-1]
                                emission_key = "emission:%s+%s"%(y0, x)
                                try:
                                    expected_counts += forward_score[index][y0] * backward_score[index+1][y1] * np.exp(w[emission_key]) * np.exp(w[transition_key])
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
        return w_score
        
    def test_gradient(self, w, path, key, value =0.01):
        gradient = crf.calculate_gradient(w, path)[key]
        w_test = copy.deepcopy(w)
        w_test[key] += value
        
        diff = (crf.calculate_loss(w_test, path) - crf.calculate_loss(w, path))
        diff /= value
        return gradient, diff
        
if __name__ == "__main__":
  crf = CRF()
  print (crf.test_gradient(crf.train_probabilities.f, "data/EN/train", 'emission:O+and',value=-.1))
  #print(crf.calculate_loss( crf.train_probabilities.f, "data/EN/train"))
  #print(crf.calculate_gradient( crf.train_probabilities.f, "data/EN/train"))