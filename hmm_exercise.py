import itertools
import json
import math
import re

def log2(p):
    if p == 0:
        return -math.inf
    else:
        return math.log2(p)

def read_model(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return HMM(data["states"], data["alphabet"], data["pi"], data["a"], data["e"])    

def log_sum(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log2(1 + 2**(b - a))
    else:
        return b + math.log2(1 + 2**(a - b))

def agreement(z1, z2):
    if len(z1) != len(z2) or len(z1) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(z1, z2) if a == b)
    return matches / len(z1)

class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        states_set = set(states)
        alphabet_set = set(alphabet)

        if set(pi.keys()) != states_set:
            raise ValueError("Pi != states")
        if set(a.keys()) != states_set:
            raise ValueError("A != states")
        for state in states:
            if set(a[state].keys()) != states_set:
                raise ValueError("A[state] != states")
            if not math.isclose(sum(a[state].values()), 1.0):
                raise ValueError("A[state] does not sum to 1")
        if set(e.keys()) != states_set:
            raise ValueError("E != states")
        for state in states:
            if set(e[state].keys()) != alphabet_set:
                raise ValueError("E[state] != alphabet")  
            if not math.isclose(sum(e[state].values()), 1.0):
                raise ValueError("E[state] does not sum to 1")

        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e

        self.log_pi = {state: log2(prob) for state, prob in pi.items()}
        self.log_a = {state: {next_state: log2(prob) for next_state, prob in a[state].items()} for state in states}
        self.log_e = {state: {symbol: log2(prob) for symbol, prob in e[state].items()} for state in states}

    def joint_probability(self, x,z):
        probability = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            probability *= self.a[z[i-1]][z[i]] * self.e[z[i]][x[i]]
        return probability
    
    def log_joint(self, x, z):
        log = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            log += self.log_a[z[i-1]][z[i]] + self.log_e[z[i]][x[i]]
        return log

    def brute_force(self, x):
        best_path = None
        max_probability = -1.0
        sum = 0.0

        for path_tuple in itertools.product(self.states, repeat=len(x)):
            z = "".join(path_tuple)
            p = self.joint_probability(x, z)
            sum += p
            if p > max_probability:
                max_probability = p
                best_path = z

        return best_path, max_probability, sum

    def viterbi(self, x):
        v = []
        back = []
        v0 = {}
        for state in self.states:
            v0[state] = self.log_pi[state] + self.log_e[state][x[0]]
        v.append(v0)
        back.append({})

        for i in range(1, len(x)):
            vi = {}
            backi = {}
            for state in self.states:
                max_score = -math.inf
                best_prev_state = None

                for prev_state in self.states:
                    score = v[i-1][prev_state] + self.log_a[prev_state][state]
                    if score > max_score:
                        max_score = score
                        best_prev_state = prev_state

                vi[state] = max_score + self.log_e[state][x[i]]
                backi[state] = best_prev_state
            v.append(vi)
            back.append(backi)

        best_last_state = None
        max_score = -math.inf

        for state, score in v[-1].items():
            if score > max_score:
                max_score = score
                best_last_state = state
        path_states = [best_last_state]
        current_state = best_last_state
        for i in range(len(back) - 1, 0, -1):
            current_state = back[i][current_state]
            path_states.append(current_state)

        path_states.reverse()
        best_path = "".join(path_states)

        return best_path, max_score


    def forward(self, x):
        f = []
        f0 = {}
        for k in self.states:
            f0[k] = self.log_pi[k] + self.log_e[k][x[0]]
        f.append(f0)

        for i in range(1, len(x)):
            fi = {}
            for k in self.states:
                log = float('-inf')
                for prev_s in self.states:
                    term = f[i-1][prev_s] + self.log_a[prev_s][k]
                    log = log_sum(log, term)
                fi[k] = log + self.log_e[k][x[i]]
            f.append(fi)

        logpx = float('-inf')
        for k in self.states:
            logpx = log_sum(logpx, f[-1][k])

        return f, logpx

    def backward(self, x):
        b = []
        bT = {}

        for k in self.states:
            bT[k] = 0.0
        b.append(bT)

        for i in range(len(x) - 2, -1, -1):
            bi = {}
            for k in self.states:
                log = float('-inf')
                for next_s in self.states:
                    term = self.log_a[k][next_s] + self.log_e[next_s][x[i+1]] + b[0][next_s]
                    log = log_sum(log, term)
                bi[k] = log
            b.insert(0, bi)

        logpx = float('-inf')
        for k in self.states:
            logpx = log_sum(logpx, self.log_pi[k] + self.log_e[k][x[0]] + b[0][k])

        return b, logpx

    def posterior(self, x):
            f, logpx = self.forward(x)
            b, _ = self.backward(x)
            n = len(x)
            post = []

            for i in range(n):
                post_i = {}
                for k in self.states:
                    log_probability = f[i][k] + b[i][k] - logpx
                    post_i[k] = 2 ** log_probability
                post.append(post_i)

            return post

    def posterior_path(self, x):
            post = self.posterior(x)
            path_chars = []

            for post_i in post:
                best_state = None
                max_prob = -1.0
                for k in self.states:
                    if post_i[k] > max_prob:
                        max_prob = post_i[k]
                        best_state = k
                path_chars.append(best_state)

            return "".join(path_chars)