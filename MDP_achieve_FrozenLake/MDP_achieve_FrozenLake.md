
# Exercise 2 - Implement Q-learning Algorithm


```python
import gym
import random
import numpy as np
```

### Q-table

In the case of FrozenLake game, we have 16 * 4 table of Q-values. We start by initializing the table to be uniform(all zeros).

    Q = np.zeros([env.observation_space.n,env.action_space.n])


```python
env = gym.make('FrozenLake-v0') #Load environment
print ("Agent Environment")
env.render()  # Output 4*4 state
rList = [] # Record reward
lr = .85
y = .99     # weight
num_episodes = 2000  #training times
Q = np.zeros([env.observation_space.n,env.action_space.n]) # Initialize Q-table
```

    Agent Environment
    
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG


### Use Q-table to make decisions `test`

* set record times(such as 4) to check move times to goals in every episode.


```python
def test(i):
    d1 = False
    j1 = 0
    start = 0
    r_sum = 0
    while d1 == False:  
        j1 +=1
        a = np.argmax(Q[start,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s1,r,d1,_ = env.step(a)
        start = s1
        r_sum +=r
    if(r_sum == 1.0):
        print ('To Goal -- Times',j1)
    else:
        print ('Not to Goal')
```

### Q-learning Algorithm

    Initialize Q(s,a) arbitrarily
    Repeat (for each episoda):
        Initialize s
        Repeat(for each step of episode):
            Choose a from s using policy derived from Q(e.g., epsilon-greedy)
            Take action a, observe r, s'
            Q(s,a) <—— Q(s,a) + alpha[reward + gamma * maxQ(s',a') - Q(s,a)]
            s <—— s'
        until s is terminal


```python
def Q_learning():
    for i in range(num_episodes):
        s = env.reset()  # Reset environment
        rAll = 0
        d = False
        j = 0

        if (i % 500) ==0:  # test Q-table
            test(i)

        while j < 99:  # Q-learning Algorithm
            j+=1
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1))) # Greedy action
            s1,r,d,_ = env.step(a)  # Obtain new state and new reward
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])  # Update Q-table
            rAll += r
            s = s1 # Update state
            if d == True:  # If agent go to the goals, break
                break

        rList.append(rAll)
```


```python
Q_learning()
print ("Accurately: " +  str(sum(rList)/num_episodes*100) + "%")
print ("Q-Table")
print (Q) # Output Q-Table
```

    Not to Goal
    To Goal -- Times 45
    To Goal -- Times 93
    Not to Goal
    Accurately: 42.35%
    Q-Table
    [[  6.94393311e-03   9.22273669e-03   7.98197631e-01   1.52587948e-02]
     [  2.54983908e-04   2.40952682e-04   1.70217715e-03   4.79265981e-01]
     [  1.87617220e-03   6.92227004e-03   2.88978635e-03   2.96276911e-01]
     [  6.74502081e-04   1.70874729e-03   4.01216360e-05   1.97994534e-01]
     [  8.81168022e-01   9.52888495e-04   7.63777537e-04   8.19136195e-04]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  4.14501407e-02   3.61209013e-04   6.48114295e-04   6.50031736e-05]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  5.96933584e-04   5.09564402e-06   2.27069141e-04   9.15041882e-01]
     [  0.00000000e+00   3.72380812e-01   1.36572750e-04   5.97140490e-04]
     [  8.81861318e-01   7.06644577e-04   0.00000000e+00   5.61846973e-04]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
     [  7.04790792e-05   4.28103812e-04   9.73697905e-01   6.24231220e-05]
     [  3.94679883e-03   9.98518477e-01   0.00000000e+00   4.03450101e-03]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]


### Results

Training for 2000 times: record Intermediate results。

**We found that the probability of finding Object is gradually increase.**

At the same time, we found that Q-table is hard to extend, because  the states of real world or other games maybe too big to describe.

![avatar](/Users/jiahuiwang/courses/machine_learning/lab/lab9/1.png)
![avatar](/Users/jiahuiwang/courses/machine_learning/lab/lab9/2.png)
![avatar](/Users/jiahuiwang/courses/machine_learning/lab/lab9/3.png)
