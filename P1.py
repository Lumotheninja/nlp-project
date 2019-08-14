from math import log
from collections import defaultdict

START_TOK = '♞START♞'
STOP_TOK = '♞STOP♞'

class TrainProbabilities:
    def __init__(self,train_path='data/EN/train'):
        self.train_path = train_path
        self._get_all_counts(train_path)
        self._get_f()

    def _get_all_counts(self,path):
        self.y_count = defaultdict(int)
        self.y0_y1_count = defaultdict(int)
        self.y_x_count = defaultdict(int)
        with open(path,mode='r',encoding="utf-8") as file:
            last_y = START_TOK
            for line in file:
                # End of a sequence
                if line=='\n':
                    y = STOP_TOK
                    self.y0_y1_count[(last_y, y)] += 1
                    last_y = START_TOK
                    self.y_count[START_TOK] += 1
                    continue

                # Extract and format word and tag
                x,y = line.split()
#                x = x.lower()

                # Increase counts
                self.y_count[y] += 1
                self.y0_y1_count[(last_y, y)] += 1
                self.y_x_count[(y, x)] += 1
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
        self.f = defaultdict(lambda:-10000.)
        self.f.update(e)
        self.f.update(q)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python P1.py <train file>")
        print ("Usage on Linux/Mac:  python3 P1.py <train file>")
        sys.exit()
    # command: python P1.py <train file>
    t = TrainProbabilities(sys.argv[1])
    print(t.f)
