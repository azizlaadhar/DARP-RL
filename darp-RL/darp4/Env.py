from logging import Logger
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import json
import math
import sys
import argparse
import shutil
import numpy as np
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from torch import nn
from transformer import Transformer
from dataset import euclidean_distance
import torch.nn.functional as f
from torch.distributions import Categorical

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


class User:
    def __init__(self):
        self.id = 0
        self.max_ride_time = 0
        self.pickup_coordinates = []
        self.dropoff_coordinates = []
        self.pickup_time_window = []
        self.dropoff_time_window = []
        self.service_duration = 0
        self.load = 0
        # Status of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served
        # 2: done
        self.status = 0
        # Flag of a user taking values in {0, 1, 2}
        # 0: waiting
        # 1: being served by the vehicle which performs an action at time step t
        # 2: done or unable to be served
        self.flag = 0
        self.served_by = 0
        self.ride_time = 0.0
    
    def relax_window(self, time_end: int):
        """drop all window constrains"""
        #earliest i can deliver
        self.dropoff_time_window[0] = 0
        #latest i can deliver
        self.dropoff_time_window[1] = time_end
        #earliest i can pickup (in order to be able to reach dropoff_time_window[0] within max ride time)
        self.pickup_time_window[0] = 0
        #latest i can pickup (in order to arrive until dropoff_time_window[1])
        self.pickup_time_window[1] = time_end
        print("setting new window for: start: %s, end: %s, for %s", self.pickup_time_window, self.dropoff_time_window, self)
    
    # def tight_window(self, time_end: int, nb_requests: int):
    #     """make the time windows as tight as possible"""
    #     #filter = lambda x: min(max(0, x), time_end)
    #     delivery_time = int(self.get_min_delivery_time())
    #     self.original_start_window = self.start_window.copy()
    #     self.original_end_window = self.end_window.copy()
    #     #outbound
    #     if self.id < nb_requests / 2:
    #         #earliest i can pickup (in order to be able to reach end_window[0] within max ride time)
    #         self.start_window[0] = max(0, self.end_window[0] - self.max_ride_time - self.service_time)
    #         #latest i can pickup (in order to arrive until end_window[1])
    #         self.start_window[1] = min(self.end_window[1] - self.service_time, time_end)
    #     #inbound
    #     else:
    #         #latest i can deliver
    #         self.end_window[0] = max(0, self.start_window[0] + self.service_time + delivery_time)
    #         #earliest i can deliver
    #         self.end_window[1] = min(self.start_window[1] + self.service_time + self.max_ride_time, time_end)
        
    #     logger.debug("setting new window for: start: %s -> %s, end: %s -> %s, for %s", self.original_start_window, self.start_window, self.original_end_window, self.end_window, self)



class Vehicle:
    def __init__(self):
        self.id = 0
        self.max_route_duration = 0
        self.max_capacity = 0
        self.route = []
        self.schedule = []
        self.ordinal = 2
        self.coordinates = []
        self.free_capacity = 0
        self.user_ride_time = {}
        self.next_free_time = 0.0
        self.service_duration = 0
        self.route_pred = [0]
        self.schedule_pred = [0]
        self.cost = 0.0


def euclidean_distance(coord_start, coord_end):
    return math.sqrt((coord_start[0] - coord_end[0]) ** 2 + (coord_start[1] - coord_end[1]) ** 2)


def time_window_shift(time_window, time):
    return [max(time_window[0] - time, 0.0), max(time_window[1] - time, 0.0)]



class darpenv():
    def __init__(self,
                 model,
                 size:int,
                 num_users:int,
                 num_vehicles:int,
                 time_end:int,
                 max_step:int,
                 max_route_duration: Optional[int]=None,
                 max_vehicle_capacity: Optional[int]=None,
                 capacity: Optional[int]=None,
                 max_ride_time: Optional[int]=None,
                 seed: Optional[int]=None,
                 window: Optional[bool]=None
                 ):
        super(darpenv, self).__init__()
        
        print("initializing env")
        self.size = size
        self.max_step = max_step
        self.num_users = num_users
        self.num_vehicles = num_vehicles
        self.max_vehicle_capcity = max_vehicle_capacity
        self.capacity = capacity
        self.time_end = time_end
        self.seed = seed
        self.current_episode = 0
        self.window = window 
        self.current_step = 0
        
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

        instance_type = parameters[args.index][1]
        num_vehicles = 2
        num_users = 16
        max_route_duration = parameters[args.index][4]
        max_vehicle_capacity = parameters[args.index][5]
        max_ride_time = parameters[args.index][6]
        print('Number of vehicles: {}.'.format(num_vehicles),
              'Number of users: {}.'.format(num_users),
              'Maximum route duration: {}.'.format(max_route_duration),
              'Maximum vehicle capacity: {}.'.format(max_vehicle_capacity),
              'Maximum ride time: {}.'.format(max_ride_time))

        
        self.model = model

        self.nodes_to_users = {}
        for i in range(1, 2 * (16 + 1) - 1):
            if i <= 16:
                self.nodes_to_users[i] = i
            else:
                self.nodes_to_users[i] = i - 16
        self.num_instance = 0
        self.obj_true = []
        self.obj_pred = []
        self.list_time_window = []
        self.list_ride_time = []
        if max_route_duration:
            self.max_route_duration = max_route_duration
        else:
            self.max_route_duration = self.max_step
        if max_ride_time:
            self.max_ride_time = max_ride_time
        else:
            self.max_ride_time = self.max_step
        self.users = []
        self.vehicles = []
        self.start_depot = np.empty(2)
        self.end_depot = np.empty(2)
        pairs=[]
        path = './instance/b2-16-test.txt'
        with open(path, 'r') as file:
            for pair in file:
                pairs.append(pair)
        self.pairs = pairs

        





    def reset (self,instance_number,relax_window=False):

        print("populate env instance with", self.num_vehicles," Vehicle and" ,self.num_users, "Users objects" )
        pair = json.loads(self.pairs[instance_number])

        num_vehicles = pair['instance'][0][0]
        num_users = pair['instance'][0][1]
        max_route_duration = pair['instance'][0][2]
        max_vehicle_capacity = pair['instance'][0][3]
        max_ride_time = pair['instance'][0][4]
        objective = pair['objective']
        #obj_true.append(objective)
        self.time_penalties =[]
        users = []
        """init users"""
        for i in range(0, num_users):
            user = User()
            user.id = i
            user.max_ride_time = max_ride_time
            user.served_by = num_vehicles
            users.append(user)
        """init nodes"""
        for i in range(0, 2 * (num_users + 1)):
            node = pair['instance'][i + 1]
            if i == 0:
                self.origin_depot_coordinates = [float(node[1]), float(node[2])]
                continue
            if i == 2 * (num_users + 1) - 1:
                self.destination_depot_coordinates = [float(node[1]), float(node[2])]
                continue
            user = users[self.nodes_to_users[i] - 1]
            if i <= num_users:
                # Pick-up nodes
                user.pickup_coordinates = [float(node[1]), float(node[2])]
                user.service_duration = node[3]
                user.load = node[4]
                user.pickup_time_window = [float(node[5]), float(node[6])]
            else:
                # Drop-off nodes
                user.dropoff_coordinates = [float(node[1]), float(node[2])]
                user.dropoff_time_window = [float(node[5]), float(node[6])]  
                vehicles = []
        """init vehicles"""
        for n in range(0, num_vehicles):
            vehicle = Vehicle()
            vehicle.id = n
            vehicle.max_capacity = max_vehicle_capacity
            vehicle.max_route_duration = max_route_duration
            vehicle.route = pair['routes'][n]
            vehicle.route.insert(0, 0)
            vehicle.route.append(2 * num_users + 1)
            vehicle.schedule = pair['schedule'][n]
            vehicle.coordinates = [0.0, 0.0]
            vehicle.free_capacity = max_vehicle_capacity
            vehicle.next_free_time = 0.0
            vehicles.append(vehicle)
        
        self.users = users
        self.vehicles = vehicles
        self.num_time_window = 0
        self.num_ride_time = []
        self.current_step = 0
        if relax_window:
            print("relaxing window constraints for all Requests")
            for request in self.users:
                request.relax_window(self.time_end)
        # else:
        #     print("tightening window constraints for all Requests")
        #     for request in self.users:
        #         request.tight_window(self.time_end, self.nb_requests)
 


        

    def get_vehicle_state(self,vehicle_id,time):
        for user in self.users:
            if user.id in self.vehicles[vehicle_id].user_ride_time.keys():
                # 1: being served by the vehicle which performs an action at time step t
                user.flag = 1
            else:
                if user.status == 0:
                    if user.load <= self.vehicles[vehicle_id].free_capacity:
                        # 0: waiting
                        user.flag = 0
                    else:
                        # 2: unable to be served
                        user.flag = 2
                        #print("canceled user ",user.id)
                else:
                    # 2: done
                    user.flag = 2
        # User information.
        users_info = [list(map(np.float64,
                               [user.service_duration,
                                user.load,
                                user.status,
                                user.served_by,
                                user.ride_time,
                                time_window_shift(user.pickup_time_window, time),
                                time_window_shift(user.dropoff_time_window, time),
                                vehicle_id,
                                user.flag]
                               + [vehicle.service_duration + euclidean_distance(
                                   vehicle.coordinates, user.pickup_coordinates)
                                  if user.status == 0 else
                                  vehicle.service_duration + euclidean_distance(
                                      vehicle.coordinates, user.dropoff_coordinates)
                                  for vehicle in self.vehicles])) for user in self.users]
        # Mask information.
        # 0: waiting, 1: being served, 2: done
        mask_info = [0 if user.flag == 2 else 1 for user in self.users]
        state = [users_info, mask_info]
        return state

    

      
    def step(self,device,state,prediction,vehicle_id):
        #print("***START ENV STEP ", self.current_step, "***")
        ustate = [user.flag for user in self.users]
        #print (ustate)

        #print("users: ", ustate)
        self.current_step +=1 
        time_penaltie =0
        
        
        vehicle = self.vehicles[vehicle_id]
        # prediction, self.log_probs[i] = self.predict(state,device)
        if prediction != self.num_users:
            user = self.users[prediction]
            if user.id not in vehicle.user_ride_time.keys():
                travel_time = euclidean_distance(vehicle.coordinates, user.pickup_coordinates)
                window_start = user.pickup_time_window[0]
                vehicle.coordinates = user.pickup_coordinates
                user.status = 1
            else:
                travel_time = euclidean_distance(vehicle.coordinates, user.dropoff_coordinates)
                window_start = user.dropoff_time_window[0]
                vehicle.coordinates = user.dropoff_coordinates
                user.status = 2
                #print("delivered user ",user.id +1)
            vehicle.cost += travel_time
            if vehicle.next_free_time + vehicle.service_duration + travel_time > window_start:
                ride_time = vehicle.service_duration + travel_time
                vehicle.next_free_time += ride_time

            else:
                ride_time = window_start - vehicle.next_free_time
                vehicle.next_free_time = window_start

            for key in vehicle.user_ride_time:
                vehicle.user_ride_time[key] += ride_time
                self.users[key].ride_time += ride_time
                if self.users[key].ride_time - self.users[key].service_duration > self.max_ride_time + 1e-6:
                    if self.users[key].id >= self.num_users / 2 or self.users[key].id + 1 != vehicle.route_pred[-1]:
                        print('The ride time of User {} is too long: {:.2f} > {:.2f}.'.format(
                            self.users[key].id + 1, self.users[key].ride_time - self.users[key].service_duration, self.max_ride_time))
                        self.num_ride_time.append(self.users[key].id + 1)
                        """give penalties for broken ride_time windows"""
                        time_penaltie += self.users[key].ride_time - self.users[key].service_duration - self.max_ride_time
                        self.time_penalties.append(time_penaltie)    
            if user.id not in vehicle.user_ride_time.keys():
                if vehicle.next_free_time < user.pickup_time_window[0] or \
                        vehicle.next_free_time > user.pickup_time_window[1]:
                    print('The pick-up time window of User {} is broken: {:.2f} not in [{}, {}].'.format(
                        user.id + 1, vehicle.next_free_time, user.pickup_time_window[0], user.pickup_time_window[1]))
                    self.num_time_window += 1
                    """give penalties for broken pickup_time windows"""
                  ### still need to implement the wait if next_free_time < user.pickup_time_window[0]
                    time_penaltie += vehicle.next_free_time-user.pickup_time_window[1]
                    self.time_penalties.append(time_penaltie)
                vehicle.user_ride_time[user.id] = 0.0
                vehicle.free_capacity -= user.load
                user.served_by = vehicle.id
            else:
                if vehicle.next_free_time < user.dropoff_time_window[0] or \
                        vehicle.next_free_time > user.dropoff_time_window[1]:
                    print('The drop-off time window of User {} is broken: {:.2f} not in [{}, {}].'.format(
                        user.id + 1, vehicle.next_free_time, user.dropoff_time_window[0], user.dropoff_time_window[1]))
                    self.num_time_window += 1
                    """give penalties for broken dropoff_time windows"""
                    time_penaltie += vehicle.next_free_time-user.dropoff_time_window[1]
                    self.time_penalties.append(time_penaltie)
                del vehicle.user_ride_time[user.id]
                vehicle.free_capacity += user.load
                user.served_by = self.num_vehicles
            user.ride_time = 0.0
            vehicle.service_duration = user.service_duration
        else:
            vehicle.cost += euclidean_distance(vehicle.coordinates, self.destination_depot_coordinates)
            vehicle.next_free_time = self.max_route_duration
            vehicle.coordinates = self.destination_depot_coordinates
            vehicle.service_duration = 0
        vehicle.route_pred.append(prediction.item() + 1)
        vehicle.schedule_pred.append(vehicle.next_free_time)
        users_delivered =0 
        for user in self.users:
            if user.flag==2:
                users_delivered+=1    


        return sum(self.time_penalties)

    def finish (self, num_finish):
        if num_finish == self.num_vehicles: #tochange
            for vehicle in self.vehicles:
                print('-> Vehicle {}'.format(vehicle.id))
                for index, node in enumerate(vehicle.route):
                    if 0 < node < 2 * self.num_users + 1:
                        vehicle.route[index] = self.nodes_to_users[node]
                ground_truth = zip(vehicle.route[1:-1], vehicle.schedule[1:-1])
                prediction = zip(vehicle.route_pred[1:-1], vehicle.schedule_pred[1:-1])
                print('Ground truth:', [term[0] for term in ground_truth])
                print('Prediction:', [term[0] for term in prediction])
                # print('Ground truth:', [f'({term[0]}, {term[1]:.2f})' for term in ground_truth])
                # print('Prediction:', [f'({term[0]}, {term[1]:.2f})' for term in prediction]
            self.obj_pred.append(sum(vehicle.cost for vehicle in self.vehicles))
            self.list_time_window.append(self.num_time_window)
            print(self.num_ride_time, len(set(self.num_ride_time)))
            self.list_ride_time.append(len(set(self.num_ride_time)))
            print('-> Objective')
            # print('Ground truth: {:.4f}.'.format(self.obj_true[-1]))
            # print('Prediction: {:.4f}.\n'.format(self.obj_pred[-1]))
        return num_finish == self.num_vehicles 

    def get_cost(self): 
        return sum(vehicle.cost for vehicle in self.vehicles)    

def main():
    args = parse_arguments()
    instance_type = parameters[args.index][1]
    num_vehicles = parameters[args.index][2]
    num_users = parameters[args.index][3]
    max_route_duration = parameters[args.index][4]
    max_vehicle_capacity = parameters[args.index][5]
    max_ride_time = parameters[args.index][6]
    print('Number of vehicles: {}.'.format(num_vehicles),
          'Number of users: {}.'.format(num_users),
          'Maximum route duration: {}.'.format(max_route_duration),
          'Maximum vehicle capacity: {}.'.format(max_vehicle_capacity),
          'Maximum ride time: {}.'.format(max_ride_time))
 
    env = darpenv(size =10 , num_users=16, num_vehicles=2, time_end=1400, max_step=100)



if __name__ == "__main__":
    main()