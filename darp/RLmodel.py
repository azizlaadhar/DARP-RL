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
from numpy import mean
from Env import darpenv
from transformer import Transformer
import time
import copy
import argparse
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
import torch.nn.functional as f

parameters = [['0', 'a', 2, 16, 480, 3, 30],  # 0
              ['1', 'a', 2, 20, 480, 3, 30],  # 1
              ['2', 'a', 2, 24, 720, 3, 30],  # 2
              ['4', 'a', 3, 24, 480, 3, 30],  # 3
              ['6', 'a', 3, 36, 720, 3, 30],  # 4
              ['9', 'a', 4, 32, 480, 3, 30],  # 5
              ['10', 'a', 4, 40, 480, 3, 30],  # 6
              ['11', 'a', 4, 48, 720, 3, 30],  # 7
              ['24', 'b', 2, 16, 480, 6, 45],  # 8
              ['25', 'b', 2, 20, 480, 6, 45],  # 9
              ['26', 'b', 2, 24, 720, 6, 45],  # 10
              ['28', 'b', 3, 24, 480, 6, 45],  # 11
              ['30', 'b', 3, 36, 720, 6, 45],  # 12
              ['33', 'b', 4, 32, 480, 6, 45],  # 13
              ['34', 'b', 4, 40, 480, 6, 45],  # 14
              ['35', 'b', 4, 48, 720, 6, 45]]  # 15


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--index', type=int, default=9)
    parser.add_argument('--mask', type=str, default='off')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    return args

def simulate_baseline(env: darpenv,device) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    nb_time_windows =0 

    while(env.finish() == False):
        free_times = [vehicle.free_time for vehicle in env.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()
        for _, vehicle_id in enumerate(indices):
            """load the current vehicle state and predict using the baselinemodel's policy"""
            state, mask = env.get_vehicle_state(vehicle_id,time,device)
            outputs = bmodel(state, device).masked_fill(mask == 0, -1e6)
            _, prediction = torch.max(f.softmax(outputs, dim=1), 1)
            reward, nb_time_window = env.step(device,state,prediction,vehicle_id,indices)
            rewards.append(reward)
            nb_time_windows =+ nb_time_window
            
    return rewards, nb_time_windows

def simulate(env: darpenv,device) -> Tuple[List[float], List[float]]:
    rewards = []
    log_probs = []
    nb_time_windows =0 

    while(env.finish() == False):
        free_times = [vehicle.free_time for vehicle in env.vehicles]
        time = np.min(free_times)
        indices = np.argwhere(free_times == time)
        indices = indices.flatten().tolist()
        for _, vehicle_id in enumerate(indices):
            """load the current vehicle state and predict using the model's policy"""
            state, mask = env.get_vehicle_state(vehicle_id,time,device)
            outputs = model(state, device).masked_fill(mask == 0, -1e6)
            _, prediction = torch.max(f.softmax(outputs, dim=1), 1)
            """load the rewards"""
            probs = nn.Softmax(dim=1)(outputs)
            m = Categorical(probs)
            log_prob = m.log_prob(prediction)
            reward, nb_time_window = env.step(device,state,prediction,vehicle_id,indices)
            rewards.append(reward)
            nb_time_windows =+ nb_time_window
            log_probs.append(log_prob)

    return rewards, log_probs, nb_time_windows



def reinforce (env,
              device,
              file,
              optimizer = torch.optim.Optimizer,
              epochs : int = 15,
              instances: int =50,
              update_baseline : int= 10,
              relax_window : bool = True):

    env.model = env.model.to(device)
    env.bmodel = env.bmodel.to(device)
    i_epoch = 0
    i_instance = 0
    mincostsRL = []
    minscoresRL = []
    broken_windowsRL = []
    costs = []
    scores = []
    broken_windows = []
    while  i_instance< instances: 
        print("----------Intance {}----------".format(i_instance))
        file.write("----------Intance {}----------".format(i_instance))
        file.write("\n")
        i_epoch = 0
        oldtrain_R =0 
        score_epochs = [2400]
        cost_epochs = [1500]
        broken_windows_epochs =[]
        while i_epoch<epochs:
            train_R = 0
            
            print('********EPOCH {}********'.format(i_epoch))
            file.write('********EPOCH {}********'.format(i_epoch))
            file.write("\n")
            objective = env.reset(i_instance)
            rewards, log_probs, nb_time_window= simulate(env ,device)
            delivered =sum([user.status!=2 for user in env.users])
            train_R =  delivered *100.0  + sum (rewards)
            print("train_R: ", train_R)
            
            score_epochs.append(train_R)
            broken_windows_epochs.append(nb_time_window)
            sum_log_probs = sum(log_probs)
            model_loss = torch.mul(-train_R , sum_log_probs)
            if (i_epoch == 0):
                model_loss = torch.mul(-train_R , sum_log_probs)
            else :
                if oldtrain_R - train_R> 0:
                    model_loss = torch.mul(oldtrain_R - train_R, sum_log_probs)
            oldtrain_R =train_R
            """ Back propagation """
            optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(env.model.parameters(), 1)
            optimizer.step()
            cost_epochs.append(env.get_cost())
            i_epoch += 1

        """TEST OLD MODEL"""
        print("ENVIRONEMNT RESET TO TEST OLD MODEL")
        env.reset(i_instance)
        with torch.no_grad():
            rewards, nb_time_window= simulate_baseline(env,device)
        train_R = sum([user.status!=2 for user in env.users])*100.0  + sum (rewards)
        cost =env.get_cost()
        
        scores.append(train_R)
        costs.append(cost-objective)
        broken_windows.append(nb_time_window)

        minscoresRL.append(min(score_epochs))
        mincostsRL.append(min(cost_epochs)-objective)
        broken_windowsRL.append(min(broken_windows_epochs))

        i_instance+=1
        print("********* END OF EPOCH *********")
        file.write("********* END OF EPOCH *********")
        file.write("\n")
        print('train_R for transformer: {}'.format( train_R))
        file.write('train_R for transformer {}: '.format( train_R))
        file.write("\n")
        print ('Min score '.format(min(score_epochs)))
        file.write('Min score {}'.format(min(score_epochs)))
        print ('Min cost diffrence'.format(min(cost_epochs)-objective))
        file.write('Min cost diffrence{}'.format(min(cost_epochs)-objective))
        print("Broken windows ",min(broken_windows_epochs))


        
    return scores, costs, minscoresRL , mincostsRL



    
if __name__ == "__main__":

    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('CUDA is available. Utilize GPUs for computation.\n')
        device = torch.device("cuda")
    else:
        print('CUDA is not available. Utilize CPUs for computation.\n')
        device = torch.device("cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--index', type=int, default=9)
    parser.add_argument('--mask', type=str, default='off')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    

    num_vehicles = 3
    num_users = 24  
    max_route_duration = 480
    max_vehicle_capacity = 6 
    max_ride_time = 45
    input_seq_len = num_users

    model = Transformer(
        num_vehicles=num_vehicles,
        num_users=num_users,
        max_route_duration=max_route_duration,
        max_vehicle_capacity=max_vehicle_capacity,
        max_ride_time=max_ride_time,
        input_seq_len=input_seq_len,
        target_seq_len=num_users + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    bmodel = Transformer(
        num_vehicles=num_vehicles,
        num_users=num_users,
        max_route_duration=max_route_duration,
        max_vehicle_capacity=max_vehicle_capacity,
        max_ride_time=max_ride_time,
        input_seq_len=input_seq_len,
        target_seq_len=num_users + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        dropout=args.dropout)

    checkpoint = torch.load('./model/model-b3-24.model')
    """LOAD THE MODEL TO OPTIMIZE"""
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    """LOAD THE OLD MODEL TO COMPARE"""
    bmodel.load_state_dict(checkpoint['model_state_dict'])
    bmodel.eval()

    """Load the log file"""
    file= open('./logsRL/logRL.txt',"w")
    env = darpenv(model,bmodel,file,size =10 , num_users=24, num_vehicles=3, time_end=1400, max_step=50)

    optimizer = torch.optim.Adam(env.model.parameters(), lr=1e-4) #1e-3

    
    scores, costs, minscoresRL, mincostsRL = reinforce(env,device,file,optimizer, epochs=2, update_baseline=10, relax_window=True)
    """SAVE THE NEW MODEL"""
    torch.save(env.model.state_dict(), './model/model-b3-24RL.model')
    path_plot = './plot/'
    os.makedirs(path_plot, exist_ok=True)
    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))
    ax1.plot(scores)
    ax1.plot(minscoresRL)
    ax1.set(xlabel="INSTANCE", ylabel="Training Penalties", title="Total Penalty")   
    plt.savefig(path_plot +'b3-24penalties.pdf')

    fig3, ax3 = plt.subplots(1,1,figsize=(10,10))
    ax3.plot(costs)
    ax3.plot(mincostsRL)
    ax3.set(xlabel="INSTANCE", ylabel="Gap with RIST", title="Cost Diffrence")  
    plt.savefig(path_plot +'b3-24gaps.pdf')

