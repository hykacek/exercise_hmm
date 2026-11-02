import itertools
import json
import math

def log2(p):
    if p == 0:
        return -math.inf
    else:
        return math.log2(p)

def read_model(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return HMM(data["states"], data["alphabet"], data["pi"], data["a"], data["e"])    

def log_sum():
    return None

def agreement():
    return None

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