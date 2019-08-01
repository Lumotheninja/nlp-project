from math import log

START_TOK = '♞START♞'
STOP_TOK = '♞STOP♞'

class TrainProbabilities:
    def __init__(self,train_path='data/EN/train'):
        self.train_path = train_path
        self._get_all_counts(train_path)
        self._get_f()
    
    def _get_all_counts(self,path):
        self.y_count = {START_TOK:0}
        self.y0_y1_count = {}
        self.y_x_count = {}
        with open(path,mode='r',encoding="utf-8") as file:
            last_y = START_TOK
            for line in file:
                # End of a sequence
                if line=='\n':
                    y = STOP_TOK
                    try:
                        self.y0_y1_count[(last_y,y)] += 1
                    except KeyError:
                        self.y0_y1_count[(last_y,y)] = 1
                    last_y = START_TOK
                    self.y_count[START_TOK] += 1
                    continue
                
                # Extract and format word and tag
                x,y = line.split()
#                x = x.lower()
                
                # Increase counts
                try:
                    self.y_count[y] += 1
                except KeyError:
                    self.y_count[y] = 1
                try:
                    self.y0_y1_count[(last_y,y)] += 1
                except KeyError:
                    self.y0_y1_count[(last_y,y)] = 1
                try:
                    self.y_x_count[(y,x)] += 1
                except KeyError:
                    self.y_x_count[(y,x)] = 1
                    
                # Update variables
                last_y = y
            
            

        
    def _get_f(self):
        e = {
            "emission:%s+%s"%(key):log(value/self.y_count[key[0]])
            for key,value in self.y_x_count.items()
        }
        q = {
            "transition:%s+%s"%(key):log(value/self.y_count[key[0]])
            for key,value in self.y0_y1_count.items()
        }
        self.f = {**e,**q}



TrainProbabilities()











