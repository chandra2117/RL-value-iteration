# VALUE ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## VALUE ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm

## VALUE ITERATION FUNCTION
### Name: CHANDRAPRIYADHARSHIINI C
### Register Number: 212223240019
```
pip install git+https://github.com/mimoralea/gym-walk #egg=gym-walk
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
envdesc  = ['SHFH','FFHF','HGFH', 'FFHF']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 9 #Enter the Goal state
P = env.env.P
P
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    # Write your code here
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob, next_state, reward, done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
# Finding the optimal policy
V_best_v, pi_best_v = value_iteration(P, gamma=0.99)
# Printing the policy
print("Name: CHANDRAPRIYADHARSHINI C")
print("Register number: 212223240019")
print()
print('Optimal policy and state-value function (VI):')
print_policy(pi_best_v, P)
# printing the success rate and the mean return
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_best_v, goal_state=goal_state)*100,
    mean_return(env, pi_best_v)))
# printing the state value function
print_state_value_function(V_best_v, P, prec=4)
```

## OUTPUT:

<img width="612" height="219" alt="Screenshot 2025-09-13 112434" src="https://github.com/user-attachments/assets/4ab52c07-a2d2-405f-a0bf-d305fa03f94c" />

<img width="769" height="58" alt="image" src="https://github.com/user-attachments/assets/d521d0c9-a972-47c1-a6b8-abb58cb16b6a" />

## RESULT:

Thus the program was successfully executed.
