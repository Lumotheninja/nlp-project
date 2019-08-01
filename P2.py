from P1 import TrainProbabilities, START_TOK, STOP_TOK
import os


class CRF:
    def __init__(self,train_path='data/EN/train'):
        self.train_path = train_path
        self.train_probabilities = TrainProbabilities(train_path)
        
        
    def apply_viterbi(self,test_path='data/EN/dev.in',save_path='data/EN/dev.p2.out'):
        try:
            os.remove(save_path)
        except:
            pass
        
        with open(test_path,mode='r',encoding="utf-8") as file:
            running_x = []
            with open(save_path,mode='w',encoding='utf-8') as write_file:
                for line in file:
                    # End of a sequence
                    if line=='\n':
                        
                        pred = self._viterbi(running_x)
                        
                        pred = pred[1:-1] # Remove start and stop tokens
                        
                        for word,label in zip(running_x,pred):
                            write_file.write('%s %s\n'%(word, label))
                        write_file.write('\n')
                        
                        running_x = []
                        
                    
                    # Extract and format word and tag
                    word = ''.join(line.split())
                    if len(word)>0:
                        running_x.append(word)
                
    
    
    def _score(self,x,y):
        """
        Calculates score for single word sequence x and label sequence y
        x: str
        y: str
        """
        
        # Extract counts
        x = x.split()
        y = y.split()
        
        y_x_count = {}
        
        y0_y1_count = {}
        
        last_y = START_TOK
        for curr_x,curr_y in zip(x,y):
            
            try:
                y_x_count[(curr_y,curr_x)] += 1
            except KeyError:
                y_x_count[(curr_y,curr_x)] = 1
            
            try:
                y0_y1_count[(last_y,curr_y)] += 1
            except KeyError:
                y0_y1_count[(last_y,curr_y)] = 1
            
            last_y = curr_y
        
        else:
            y0_y1_count[(last_y,STOP_TOK)] = 1
        
        
        # Extract counts of features
        e = {
            "emission:%s+%s"%(key):value
            for key,value in y_x_count.items()
        }
        q = {
            "transition:%s+%s"%(key):value
            for key,value in y0_y1_count.items()
        }
        f = {**e,**q}
        
        # Return score
        return sum([self.train_probabilities.f[key]*value for key,value in f.items()])
    
    
    def _viterbi(self,sentence):
        """
        Performs Viterbi on input sentence
        sentence: list of words
        """
        last_layer_scores = {START_TOK:0} # Stores greedy score up till nodes in last layers
        last_layer_seq = {START_TOK:[START_TOK]} # Stores greedy sequence up till nodes in last layers
        possible_y = self.train_probabilities.y_count.keys()
        
        
        for x in sentence:
            next_layer_scores = {}
            next_layer_seq = {}
            
            
            
            for next_y in possible_y:
                for last_y,last_score in last_layer_scores.items():
                    emission_key = "emission:%s+%s"%(next_y,x)
                    transition_key = "transition:%s+%s"%(last_y,next_y)
                    try:
                        emission_weight = self.train_probabilities.f[emission_key]
                    except KeyError:
                        emission_weight = 0
                        
                    
                    try:
                        transition_weight = self.train_probabilities.f[transition_key]
                    except KeyError:
                        transition_weight = 0
                        
                    
                    curr_score = last_score + emission_weight + transition_weight
                    update_flag = False
                    try:
                        if next_layer_scores[next_y] < curr_score:
                            update_flag = True
                    except KeyError:
                        update_flag = True
                    if update_flag:
                        next_layer_scores[next_y] = curr_score
                        next_layer_seq[next_y] = last_layer_seq[last_y] + [next_y]
                        
            
            last_layer_scores = next_layer_scores
            last_layer_seq = next_layer_seq
                        
        # Update final transition score
        next_y = STOP_TOK
        next_layer_scores = {}
        next_layer_seq = {}
        for last_y,last_score in last_layer_scores.items():
            transition_key = "transition:%s+%s"%(last_y,next_y)
            try:
                transition_weight = self.train_probabilities.f[transition_key]
            except KeyError:
                transition_weight = 0
            curr_score = last_score + transition_weight
            update_flag = False
            try:
                if next_layer_scores[next_y] < curr_score:
                    update_flag = True
            except KeyError:
                update_flag = True
            if update_flag:
                next_layer_scores[next_y] = curr_score
                next_layer_seq[next_y] = last_layer_seq[last_y] + [next_y]
        try:
            return next_layer_seq[STOP_TOK]
        except:
            print(last_layer_scores)
            raise Exception
    
    


crf = CRF()
crf.apply_viterbi()
#print(crf._viterbi('All in all , the food was great ( except the desserts ) .'.split()))
#print(crf.train_probabilities.f.keys())




#for y in 'O B-positive B-negative'.split():
#    try:
#        print(crf.train_probabilities.f["transition:%s+%s"%(y,STOP_TOK)])
#    except:
#        pass


"""
emission:O+All transition:♞START♞+O -9.079345204990318
emission:B-positive+All transition:♞START♞+B-positive -3.153270067770207
emission:B-negative+All transition:♞START♞+B-negative -4.539564428890097
emission:B-neutral+All transition:♞START♞+B-neutral -5.0503900526560885
"""





























        
        

        
