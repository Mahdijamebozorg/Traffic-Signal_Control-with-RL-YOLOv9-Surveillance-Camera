from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

class TrafficLightEnv(Env):
  def __init__(self, roads_count, change_size=10, max_value=500):

    self.roads_count = roads_count
    self.max_value = max_value
    self.change_size = change_size
    self.max_iter = 100
    # self.patience = roads_count+1
    self.seed = 0

    # For each road: 0:-change_size 1:0 2:+change_size and also change all
    # self.action_space = MultiDiscrete([3 for _ in range(self.roads_count + 1)])
    # self.action_space = Box(low=0, high=1, shape=(roads_count*3 + 3,), dtype=np.uint8) # Optimized for deep learning
    self.action_space = Discrete((roads_count+1)*3)

    # Observations
    # self.observation_space= Dict({
    #   'green light timer': Box(low=0, high=100, dtype=np.int16, shape=(self.roads_count,)),
    #   'avg waiting times' : Box(low=0, high=100, dtype=np.float16, shape=(self.roads_count,)),
    #   'vehicles counts' : Box(low=0, high=100, dtype=np.int16, shape=(self.roads_count,)),
    #   'in counts' : Box(low=0, high=100, dtype=np.int16, shape=(self.roads_count,)),
    #   'out counts' : Box(low=0, high=100, dtype=np.int16, shape=(self.roads_count,)),
    #   'avg speeds' : Box(low=0, high=100, dtype=np.float16, shape=(self.roads_count,)),
    # })
    self.observation_space = Box(low=0,high=max_value,shape=(16,)) # Optimized for deep learning

    # Each road timer (main state)
    self.green_light_timer = np.round(self.observation_space.sample()[0 : self.roads_count])
    # average waiting time of each road
    self.avg_waiting_times = np.array([np.sum([rt if i!=r else 0 for i,rt in enumerate(self.green_light_timer)]) for r in range(self.roads_count)])
    # vehicles average speed
    # self.avg_speeds = np.round(self.observation_space.sample()[self.roads_count*2 : self.roads_count*3])
    # vehicles stopped in each road
    self.vehicles_counts = np.round(self.observation_space.sample()[self.roads_count*2 : self.roads_count*3])
    # vehicles passed in green light of each road
    self.in_counts = np.round(self.observation_space.sample()[self.roads_count*3 : self.roads_count*4])
    # vehicles passed out of each road
    # self.out_counts = np.round(self.observation_space.sample()[self.roads_count*5 : self.roads_count*6])

    # other var if needed = self.observation_space.sample()[self.roads_count*6 : self.roads_count*7] ...

    self.state = np.reshape([
      self.green_light_timer//self.max_value,
      self.avg_waiting_times//(self.max_value*3),
      self.vehicles_counts//self.max_value,
      self.in_counts//self.max_value,
      #  self.out_counts,
      #  self.avg_speeds
       ],-1)



  def reset(self):

    # Reset environment variables to initial state for a new episode
    super().reset(seed=0)
    self.green_light_timer = np.round(self.observation_space.sample()[0 : self.roads_count])
    self.avg_waiting_times = np.array([np.sum([rt if i!=r else 0 for i,rt in enumerate(self.green_light_timer)]) for r in range(self.roads_count)])
    # self.avg_speeds = np.round(self.observation_space.sample()[self.roads_count*2 : self.roads_count*3])
    self.vehicles_counts = np.round(self.observation_space.sample()[self.roads_count*2 : self.roads_count*3])
    self.in_counts = np.round(self.observation_space.sample()[self.roads_count*3 : self.roads_count*4])
    # self.out_counts = np.round(self.observation_space.sample()[self.roads_count*5 : self.roads_count*6])
    self.max_iter = 100
    # self.patience = self.roads_count+1
    
    self.state = np.reshape([
    self.green_light_timer//self.max_value,
    self.avg_waiting_times//(self.max_value*3),
    self.vehicles_counts//self.max_value,
    self.in_counts//self.max_value,
    # self.out_counts,
    # self.avg_speeds
    ],-1)
    return self.state
  
  # when action space is flatten
  def cast_action(self, action):
    on_hot = np.zeros((self.roads_count+1)*3)
    on_hot[action] = 1
    action = np.reshape(on_hot, newshape=(self.roads_count+1,3))
    # set unchanged max index to 1
    for act in action:
       if np.sum(act) == 0:
          act[1] = 1
    return np.argmax(action, axis=1)

  def render(self):
        print('green light timer', self.green_light_timer)
        print('avg waiting times (-)', self.avg_waiting_times)
        print('vehicles counts (-)', self.vehicles_counts)
        print('in counts (+)', self.in_counts)
        # print('out counts (+)', self.out_counts)
        # print('avg speeds (+)', self.avg_speeds)


  # action is a list of one-hot encoded action for each road (road_count*3)
  def step(self, action):

    action = self.cast_action(action)

    # apply action on each raod green light timer: 0:-10 1:0 2:+10
    for r in range(self.roads_count):
        # decrease
        if(action[r]==0):
            if(self.green_light_timer[r]>self.change_size):
                self.green_light_timer[r] -= self.change_size
        # increase
        elif(action[r]==2):
            if(self.green_light_timer[r] + self.change_size <= self.max_value):
                self.green_light_timer[r] += self.change_size
        else:
            pass

    # change all green timers (increase or decrease if not causes negative)
    # decrease
    if(action[-1]==0):
        # if all timer are bigger that dec value
        if(all(i > self.change_size for i in self.green_light_timer)):
            self.green_light_timer = np.subtract(self.green_light_timer, self.change_size*2)
    # increase
    elif(action[-1]==2):
        if(all(i + self.change_size <= self.max_value for i in self.green_light_timer)):
            self.green_light_timer = np.add(self.green_light_timer, self.change_size)
    else:
        pass

    # ---------------------------- #
    #  Simulating input variables  #
    # ---------------------------- #

    # calcualte red ligh time based on other roads green light timer
    for r in range(self.roads_count):
      self.avg_waiting_times[r] = np.sum([rt if i!=r else 0 for i,rt in enumerate(self.green_light_timer)])

    # calcualte vehicles passed in the green light
    for r in range(self.roads_count):
      # if road is not empty
      if self.vehicles_counts[r] > 0:
          # if green time is high, all vehicles pass and if is low, 1 vehicle per time passes
          self.in_counts[r] = min(self.vehicles_counts[r], self.green_light_timer[r])
      else:
         self.in_counts[r] = 0

    # calcualte vehicles passed out
    # rands = np.random.randint(0,self.in_counts.sum(),self.roads_count)
    # rands = np.round((rands/np.sum(rands))*self.in_counts.sum()) # make sure sum of the is sum of inputs
    # self.out_counts = rands.copy()
    
    # vehicles not passed
    remaining_vehicles = self.vehicles_counts.copy()

    # calculate vehicles count 
    for r in range(self.roads_count):
      self.vehicles_counts[r] -= self.in_counts[r]
      remaining_vehicles[r] = self.vehicles_counts[r]
      if self.max_value - self.vehicles_counts[r] > 0:
          self.vehicles_counts[r] += np.random.randint(0, self.max_value - self.vehicles_counts[r] + 1)


    # ---------------------- #
    #          Done          #
    # ---------------------- #

    # removed according to model desire to finish soon 
    # stable states counter
    # if all are unchaged
    # if(all([act==1 for act in action])):
    #    self.patience -= 1
    # else:
    #    self.patience = self.roads_count+1 # reset

    # check iterations
    self.max_iter -= 1
    done = True if self.max_iter==0 else False

    # removed according to model desire to finish soon 
    # check patience
    # if self.patience == 0:
    #    done = True
  


    # ------------------------ #
    #          Reward          #
    # ------------------------ #

    reward = 0
    # Penalties
    reward -= self.avg_waiting_times.sum()
    reward -=  remaining_vehicles.sum()
    # Fairness (wighted)
    reward -= (np.max(self.avg_waiting_times) - np.min(self.avg_waiting_times))*2
    # Reward
    reward += self.in_counts.sum()
    # reward += self.out_counts.sum()
    # reward += self.avg_speeds.sum()

    # normalize, only for this case
    reward /= (self.max_value * (12+4+12+4)) 

    # normalize, only for this case
    self.state = np.reshape([
    self.green_light_timer//self.max_value,
    self.avg_waiting_times//(self.max_value*3),
    self.vehicles_counts//self.max_value,
    self.in_counts//self.max_value,
    # self.out_counts,
    # self.avg_speeds
    ],-1)

    info = {}

    # Return new state, reward and info dictionary
    return self.state, reward, done, info
