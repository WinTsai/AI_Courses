# *-utf-8-*
# Markov algrithom implementation
# Date: 2020.08.25
# Author: Tsai

import numpy as np
import random as rm
#from sympy import limit
# states space
states = ['c1','c2','c3']

# possible events squence
transitionName = [['c11','c12','c13'],['c21','c22','c23'],['c31','c32','c33']]
# transition posibility matrix
transitionMatrix = [[0.3,0.2,0.5],[0.5,0.3,0.2],[0.2,0.5,0.3]]

if (sum(transitionMatrix[0]) !=1) or(sum(transitionMatrix[1]) != 1) or (sum(transitionMatrix[2])!=1):
    print('transition Matrix perhaps wrong!')
else:
    print('All goes well!')

# n step transiton states and posibility
def state_trans_forecast(initState, steps):
    #initial states: c1,c2 or c3
    print("Start state: " + initState)
    #record the initial state
    stateList = [initState]
    i = 0
    # 计算 stateList 的概率
    prob = 1
    while i != steps:
        if initState == 'c1':
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == 'c11':
                prob = prob * 0.3
                stateList.append('c1')
                pass
            elif change == 'c12':
                prob = prob * 0.2
                initState = 'c2'
                stateList.append('c2')
            else:
                prob = prob * 0.5
                initState = 'c13'
                stateList.append('c3')
        elif initState == 'c2':
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == 'c22':
                prob = prob * 0.3
                stateList.append('c2')
                pass
            elif change == 'c21':
                prob = prob * 0.2
                initState = 'c1'
                stateList.append('c1')
            else:
                prob = prob * 0.5
                initState = 'c23'
                stateList.append('c3')
        elif initState == 'c3':
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == 'c33':
                prob = prob * 0.3
                stateList.append('c31')
                pass
            elif change == 'c1':
                prob = prob * 0.5
                initState = 'c1'
                stateList.append('c1')
            else:
                prob = prob * 0.2
                initState = 'c32'
                stateList.append('c2')
        i += 1  
    print("Possible states: " + str(stateList))
    print("End state after "+ str(steps) + " steps: " + initState)
    print("Probability of the possible sequence of states: " + str(prob))


def MatrixNpower(Mtx,steps):
    if(type(Mtx)==list):
        Mtx=np.array(Mtx)
    if(steps==1):
        return Mtx
    else:
        return np.matmul(Mtx,MatrixNpower(Mtx,steps-1))
# forcast the state after 2 steps
state_trans_forecast('c2',3)

# N steps transition matrix
N2 = MatrixNpower(transitionMatrix,2)
print(N2)