from logging import Logger
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
from torch import optim
import torch.nn as nn
import torch
from typing import Tuple, List
from torch.distributions import Categorical
import numpy as np
from Env import darpenv
import time
import copy




def simulate(max_step: int, env: darpenv,device , greedy: bool=False ) -> Tuple[List[float], List[float]]:
    

    rewards = []
    log_probs = []
    
    for t in range(max_step):
        state = env.get_state()
        reward, log_prob,finished = env.step(env,device,state)
        rewards.append(reward)
        log_probs.append(log_prob)
        if finished:
             break 
    return rewards , log_probs



def reinforce (env,
              optimizer = torch.optim.Optimizer,
              epochs : int = 100,
              max_step : int= 100,
              update_baseline : int= 10,
              relax_window : bool = True,
              device= torch.device('cuda')):

    baseline = copy.deepcopy(env.model)
    baseline = baseline.to(device)
    model = model.to(device)
    scores = []
    tests = []
    train_R = 0
    baseline_R = 0
    for i_epoch in  range(epochs):
        print("*** EPOCH %s ***", i_epoch)
        for i_episode in env.max_steps:
            print(i_episode)
            """update baseline model after every 500 steps"""
            if i_episode % update_baseline == 0:
                if train_R >= baseline_R:
                    print("new baseline model selected after achiving %s reward", train_R)
                    baseline.load_state_dict(model.state_dict())
            path = './instance/b2-16-test.txt'
            env.reset(path)
            baseline_env = copy.deepcopy(env)
            
            """ Simulate episode with train and baseline model """
            with torch.no_grad():
               baseline_rewards, _ = simulate(max_step, baseline_env, baseline,device, greedy= True)
            rewards, log_probs= simulate(max_step, env, model ,device)
            
            """Aggregate rewards"""
            train_R = sum(rewards)
            baseline_R = sum(baseline_rewards)
            sum_log_probs = sum(log_probs)
            scores.apppend(train_R)
            model_loss = torch.mean (-train_R * sum_log_probs)
            """ Back propagation """
            optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if i_episode % 100 == 0:
                env.reset(relax_window)
                with torch.no_grad():
                    rewards, log_probs = simulate(max_step, env, model,device, greedy=True)
                delivered = sum([user.flag==2 for user in env.users])
                print('Episode: {}, total distance: {:.2f}'.format(i_episode, sum(rewards)))
                tests.append((sum(rewards), delivered))
                baseline.load_state_dict(model.state_dict()) #update baseline
                delivering = sum([user.flag ==1 for user in env.users])
                pickup = sum([user.flag==0 for user in env.users])
                print(f'delivered: {delivered}, delivering: {delivering}, waiting: {pickup}')
    #TODO: create result object
    return scores, tests



    
if __name__ == "__main__":
    env = darpenv(size =10 , num_users=16, num_vehicles=2, time_end=1400, max_step=100)
    env.reset('./instance/b2-16-test.txt')

    optimizer = torch.optim.Adam(env.model.parameters(), lr=1e-3)

    
    scores, tests = reinforce(env,optimizer, epochs=100, max_step=100, update_baseline=500, relax_window=True)
    # dump_data(scores, "models/scores.pkl")
    # dump_data(tests, "models/tests.pkl")
    PATH = "models/test.pth"
    torch.save(env.model.state_dict(), PATH)
    path_plot = './plot/'
    os.makedirs(path_plot, exist_ok=True)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(scores)
    ax.set(xlabel="Epsidoe", ylabel="Training Reward", title="Total distance")      
    plt.savefig(path_plot +'b2-16.pdf')
    #TEST ENV WITH LOG
    print("FINAL TEST WITH LOGS")
    env.reset('./instance/b2-16-test.txt')
    rewards, _ = simulate(100, env, env.model)
    rewards = sum([r.state == "delivered" for r in env.users])
    print(f"total delivered: {rewards}")
