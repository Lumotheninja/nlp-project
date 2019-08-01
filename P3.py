from P1 import TrainProbabilities, START_TOK, STOP_TOK
from P2 import CRF
import numpy as np

class CRF(CRF):
    def _forward(self, w, sentence):
        possible_y = self.train_probabilities.y_count.keys()

        alpha_list = [{START_TOK: 1}]

        # START to all y
        last_y = START_TOK
        alpha_score = {}
        x = sentence[0]

        for next_y in possible_y:
            
            transition_key = "transition:%s+%s"%(last_y,next_y)
            emission_key = "emission:%s+%s"%(next_y, x)
            try:
                alpha_score[next_y] = np.exp(w[transition_key]) * np.exp(w[emission_key])
            except KeyError:
                alpha_score[next_y] = 0
        alpha_list.append(alpha_score)
        
        # all y to all y
        for index, x in enumerate(sentence[1:]):
            alpha_score = {}
            for next_y in possible_y:
                sum_prob = 0
                for last_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    try:
                        sum_prob += np.exp(w[transition_key]) * alpha_list[index+1][last_y]
                    except KeyError:
                        pass
                emission_key = "emission:%s+%s"%(next_y, x)
                try:
                    alpha_score[next_y] = np.exp(w[emission_key]) * sum_prob
                except KeyError:
                    alpha_score[next_y] = 0
            alpha_list.append(alpha_score)
        
        
        # all y to STOP
        next_y = STOP_TOK
        alpha_score = {}
        sum_prob = 0
        for last_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                sum_prob += np.exp(w[transition_key]) * alpha_list[-1][last_y]
            except KeyError:
                pass
        alpha_score[next_y] = sum_prob
        alpha_list.append(alpha_score)

        return alpha_list

    def calculate_loss(self, w, x, y):
        forward_score = self._forward(w, x)
        # log the score stored at the last element of forward score 
        return np.log(forward_score[-1]["STOP"]) - self._score(x, y)

    def _backward(self, w, sentence):
        
        possible_y = self.train_probabilities.y_count.keys()

        beta_list = [{STOP_TOK: 1}]

        # all y to STOP
        next_y = STOP_TOK
        beta_score = {}
        for last_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                beta_score[last_y] = np.exp(w[transition_key])
            except KeyError:
                beta_score[last_y] = 0
        beta_list.append(beta_score)
        
        # all y to all y
        for index, x in enumerate(sentence[1:][::-1]):
            beta_score = {}
            for last_y in possible_y:
                sum_prob = 0
                for next_y in possible_y:
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    emission_key = "emission:%s+%s"%(next_y, x)
                    try:
                        sum_prob += np.exp(w[transition_key]) * np.exp(w[emission_key]) * beta_list[index+1][next_y]
                    except KeyError:
                        pass
                beta_score[last_y]  = sum_prob
            beta_list.append(beta_score)
        
        # START to all y
        last_y = START_TOK
        beta_score = {}
        sum_prob = 0
        x = sentence[0]
        for next_y in possible_y:
            transition_key = "transition:%s+%s"%(last_y,next_y)
            emission_key = "emission:%s+%s"%(next_y, x)
            try:
                sum_prob += np.exp(w[transition_key]) * np.exp(w[emission_key]) * beta_list[-1][next_y]
            except KeyError:
                pass
        beta_score[last_y] = sum_prob
        beta_list.append(beta_score)

        return beta_list[::-1]
                
    def _gradient(self, x, y):
        """
        Calculates gradient for single word sequence x and label sequence y
        x: str
        y: str
        """
        x = x.split('')
        y = y.split('')
        sent_len = len(x)
        tag_len = len(self.train_probabilities.y_count.keys())
        w = crf.train_probabilities.f
        
        forward = crf._forward(w, x)
        backward = crf._backward(w, x)
        denom = forward[-1][STOP_TOK]
        
        w_score = {}

        for weights in w.keys():

            # emission score
            if weights[0] == 'e':
                next_y = weights.split(":")[1].split("+")[0]
                word = weights.split(":")[1].split("+")[0]
                gradient_score = 0

                # sum over all possible last y that can go to next y and next y to word
                for index in range(len(forward)-1):
                    for last_y in forward[index].keys():
                        transition_key = "transition:%s+%s"%(last_y,next_y)
                        try:
                            gradient_score += forward[index][last_y] * w[transition_key] * backward[index+1][next_y] * w[weights]
                    except KeyError:
                        pass
                
                gradient_score /= denom

                # minus off count, calculated using closed form formula
                gradient_score -= (tag_len**(sent_len-2) * (sent_len - 1))
                w_score[weights] = gradient_score
                
            # transmission score
            else:
                last_y = weights.split(":")[1].split("+")[0]
                next_y = weights.split(":")[1].split("+")[0]
                gradient_score = 0

                # sum over all possible indices
                for index in range(len(forward)-1):
                    try:
                        gradient_score += forward[index][last_y] * w[weights] * backward[index+1][next_y]
                    except KeyError:
                        pass
                
                gradient_score /= denom

                # minus off count, calculated using closed form formula
                gradient_score -= (tag_len**(sent_len-2) * (sent_len - 1))
                w_score[weights] = gradient_score
            
            return w_score
            
    def apply_gradient(self,train_path='data/EN/train'):
        try:
            os.remove(save_path)
        except:
            pass
        
        pass
        


crf = CRF()

































        
        

        
