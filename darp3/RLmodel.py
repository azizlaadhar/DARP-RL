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



def simulate(max_step: int, env: darpenv,device ) -> Tuple[List[float], List[float]]:
    

    rewards = []
    log_probs = []
    num_finish = 0
    t=0
    while(t<50):
        #print (len(vehicles_available))
        t+=1
        if len(env.vehicles_available)==0 or num_finish==2:
            break
        next_free_times = [vehicle.next_free_time for vehicle in env.vehicles_available]
        time = np.min(next_free_times)
        indices = np.argwhere((next_free_times == time))
        indices = indices.flatten().tolist() 
        num_finish =0
        for vehicle in env.vehicles_available:
            print(vehicle.route_pred)
            env.get_vehicle_state(vehicle.id,time)
            if vehicle.next_free_time > env.max_route_duration:
                num_finish += 1
            
        
      
        for _, vehicle_id in enumerate(indices):
            state, _ = env.get_vehicle_state(vehicle_id,time)
            #state, _ = DataLoader([state, 0], batch_size=1)  # noqa
            outputs = model(DataLoader([state, 0], batch_size=1) , device)
            _, prediction = torch.max(f.softmax(outputs, dim=1), 1)
            #input(prediction)
            probs = nn.Softmax(dim=1)(outputs)
            m = Categorical(probs)
            #action = m.sample()
            log_prob = m.log_prob(prediction)
            reward = env.step(device,state,prediction,vehicle_id)
            rewards.append(reward)
            log_probs.append(log_prob)
            print(vehicle_id)
        
        env.get_vehicles_available(time)

        print (env.vehicles[0].next_free_time)
        print (env.vehicles[1].next_free_time)
    return rewards , log_probs



def reinforce (env,
              device,
              optimizer = torch.optim.Optimizer,
              epochs : int = 20,
              max_step : int= 100,
              update_baseline : int= 10,
              relax_window : bool = True):

    baseline = copy.deepcopy(env.model)
    baseline = baseline.to(device)
    env.model = env.model.to(device)
    scores = []
    tests = []
    costs = []
    train_R = 0
    baseline_R = -20000
    i_epoch = 0
    while  i_epoch< epochs: 
        train_R = 0
        print('********EPOCH {}********'.format(i_epoch))
        env.reset(i_epoch)
        """update baseline model after every 500 steps"""
        if epochs % update_baseline == 9:
            if train_R >= baseline_R:
                print("new baseline model selected after achiving ", train_R, " reward")
                baseline.load_state_dict(env.model.state_dict())
        #env.reset(i_epoch)
        baseline_env = copy.deepcopy(env)
        
        """ Simulate episode with train and baseline model """
        with torch.no_grad():
           #input("Simulate baseline model")
           baseline_rewards, _ = simulate(max_step, baseline_env,device)
         
        env.reset(i_epoch)
        rewards, log_probs= simulate(max_step, env ,device)

        i_epoch += 1
        """Aggregate rewards"""
        """-sum(rewards) +"""
        print("delivered reward ", sum([user.delivered for user in env.users]))
        
        train_R =    sum(rewards) - sum([(user.delivered==False) for user in env.users])*10
        print("train_R: ", train_R)
        baseline_R = -sum(baseline_rewards)
        print("baseline_R: ",baseline_R)
        sum_log_probs = sum(log_probs)
        scores+=[train_R]

        model_loss = torch.mean (torch.mul(train_R , sum_log_probs))
        """ Back propagation """


        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(env.model.parameters(), 1)
        optimizer.step()

        
        print("ENVIRONEMNT RESET")
        env.reset(i_epoch)#relax_window
        #input("simulate eval model")
        with torch.no_grad():
            rewards, log_probs = simulate(max_step, env,device)
        delivered = sum([user.flag==2 for user in env.users])
        print('Episode: {}, total distance: {:.2f}'.format(epochs, sum(rewards)))
        tests.append((-sum(rewards), delivered))
        costs.append(env.get_cost())
        #baseline.load_state_dict(env.model.state_dict()) #update baseline
        delivering = sum([user.flag ==1 for user in env.users])
        pickup = sum([user.flag==0 for user in env.users])
        print(f'delivered: {delivered}, delivering: {delivering}, waiting: {pickup}')
    #         if i_episode % 50 == 0:
    #             print("ENVIRONEMNT RESET")
    #             env.reset(i_epoch,relax_window=True)#relax_window
    #             input("simulate eval model")
    #             with torch.no_grad():
    #                 rewards, log_probs = simulate(max_step, env,device)
    #             delivered = sum([user.flag==2 for user in env.users])
    #             print('Episode: {}, total distance: {:.2f}'.format(i_episode, sum(rewards)))
    #             tests.append((-sum(rewards), delivered))
    #             baseline.load_state_dict(env.model.state_dict()) #update baseline
    #             delivering = sum([user.flag ==1 for user in env.users])
    #             pickup = sum([user.flag==0 for user in env.users])
    #             print(f'delivered: {delivered}, delivering: {delivering}, waiting: {pickup}')
    # #TODO: create result object
        
    return scores, tests, costs



    
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
    model = Transformer(
            num_vehicles=2,
            num_users=16,
            max_route_duration=480,
            max_vehicle_capacity=6,
            max_ride_time=45,
            input_seq_len=16,
            target_seq_len=16 + 1,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            dropout = args.dropout)
        
    checkpoint = torch.load('./model/model-b2-16.model')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    env = darpenv(model,size =10 , num_users=16, num_vehicles=2, time_end=1400, max_step=50)


    optimizer = torch.optim.Adam(env.model.parameters(), lr=1e-3)

    
    scores, tests, costs = reinforce(env,device,optimizer, epochs=10, max_step=50, update_baseline=10, relax_window=True)
    # dump_data(scores, "models/scores.pkl")
    # dump_data(tests, "models/tests.pkl")
    
    #torch.save(env.model.state_dict(), path_plot)
    path_plot = './plot/'
    os.makedirs(path_plot, exist_ok=True)
    fig1, ax1 = plt.subplots(1,1,figsize=(10,10))
    #ax1.plot(scores)
    ax1.plot(tests)
    ax1.set(xlabel="EPOCHS", ylabel="Rewards", title="Total rewards")   
    plt.savefig(path_plot +'b2-16penal.pdf')


    fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    #ax2.plot(scores)
    ax2.plot(costs)
    ax2.set(xlabel="EPOCHS", ylabel="Testing Costs", title="Total Cost")  
    plt.savefig(path_plot +'b2-16costal.pdf')
    #TEST ENV WITH LOG
    # print("FINAL TEST WITH LOGS")
    # env.reset(50)
    # rewards, _ = simulate(100, env, device)
    # rewards = sum([r.state == "delivered" for r in env.users])
    # print(f"total delivered: {rewards}")
