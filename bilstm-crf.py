import torch.nn as nn
import torch.optim as optim
import torch

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Map output of LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters. Entry i, j is the score of transitioning to i from j
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # Enforce constraint to never transition to start tag and transit from stop tag
        self.transitions.data[tag_to_ix[START_TOK], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TOK]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))
    
    def _forward_algo(self, feats):
        # Forward algorithm to compute partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TOK]] = 0

        forward_var = init_alphas

        # Iterate through sentence
        for feat in feats:
            alphas_t = []  # forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast emission score
                emission_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # ith entry of transition_score is the score of transitioning to next_tag from i
                transition_score = self.transitions[next_tag].view(1, -1)
                # ith entry of next_tag_var is the value for (i -> next_tag) before doing log-sum-exp
                next_tag_var = forward_var + transition_score + emission_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TOK]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeddings = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TOK]], dtype=torch.long), tags])
        for i , feat in enumerate(feats):
            score += self.transitions[tags[i+1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_ix[STOP_TOK], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []

        init_viterbi = torch.full((1, self.tagset_size), -10000.)
        init_viterbi[0][self.tag_to_ix[START_TOK]] = 0

        forward_var = init_viterbi
        for feat in feats:
            bp_t = []  # backpointers for this step
            virterbi_vars_t = []  # viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # max does not depend on emission scores, will add that later
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bp_t.append(best_tag_id)
                virterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1))
            # add emission scores
            forward_var = (torch.cat(virterbi_vars_t) + feat).view(1, -1)
            backpointers.append(bp_t)
        
        # transition to STOP
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TOK]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Retrace backpointers to decode best path
        best_path = [best_tag_id]
        for bp_t in reversed(backpointers):
            best_tag_id = bp_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TOK]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_algo(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if to_ix.get(w) else len(to_ix)-1 for w in seq ]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def parse_training_file(path):
    training_data = []
    with open(path, mode='r', encoding="utf-8") as f:
        last_y = START_TOK
        running_x = []
        running_y = []
        for line in f:
            # End of a sequence
            if line=='\n':
                training_data.append((running_x, running_y))
                running_x = []
                running_y = []
                continue

            # Extract and format word and tag
            x,y = line.split()
            running_x.append(x)
            running_y.append(y)
    return training_data

def parse_test_file(path):
    test_data = []
    with open(path, mode='r', encoding='utf-8') as f:
        running_x = []
        for line in f:
            if line == '\n':
                test_data.append(running_x)
                running_x = []
                continue
            running_x.append(line.rstrip('\n'))
    return test_data


def train(f_path):
    training_data = parse_training_file(f_path)

    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix["UNK"] = len(word_to_ix)
    model = BiLSTMCRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # plus one for UNKNOWN
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CHeck predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[5][0], word_to_ix)
        print(model(precheck_sent))
    
    for epoch in range(EPOCHS):
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'bilstm_model_ES.pt')
    
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[5][0], word_to_ix)
    #     print(model(precheck_sent))

def test(f_path, out_path):
    # training_data = parse_training_file('data/EN/train')

    # for sentence, tags in training_data:
    #     for word in sentence:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    # word_to_ix["UNK"] = len(word_to_ix)
    model = BiLSTMCRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load('bilstm_model_ES.pt'))
    model.eval()

    ix_to_tag = {v:k for k,v in tag_to_ix.items()}

    test_data = parse_test_file(f_path)
    output = []
    for sentence in test_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        _, tags_ix = model(sentence_in)
        tags = [ix_to_tag[t] for t in tags_ix]
        output.append((sentence, tags))

    with open(out_path, mode="w", encoding='utf-8') as f:
        for sentence, tags in output:
            for word, tag in zip(sentence, tags):
                f.write(f"{word} {tag}\n")
            f.write("\n")

if __name__ == "__main__":
    START_TOK = '♞START♞'
    STOP_TOK = '♞STOP♞'
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10
    word_to_ix = {}
    # TAG to IDX for EN
    # tag_to_ix = {'O': 0, 'B-positive': 1, 'B-negative': 2, 'I-positive': 3, 'B-neutral': 4, 'I-neutral': 5, 'I-negative': 6, START_TOK: 7, STOP_TOK: 8}
    # TAG to IDX for ES
    tag_to_ix = {'O': 0, 'B-positive': 1, 'B-negative': 2, 'I-positive': 3, 'B-neutral': 4, 'I-neutral': 5, 'I-negative': 6, 'B-conflict': 7, START_TOK: 8, STOP_TOK: 9}
    train('data/ES/train')
    test('data/ES/dev.in', 'data/ES/dev.p5.out')

