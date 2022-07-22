import numpy as np
import json
import os
import sys
from env import make_env
from Models.LunarControllerModel import make_controller, simulateLunar
from es import CMAES
import time

def initialize_settings(sigma_init=0.1):
  global population, filebase, controller, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
  population = num_worker*num_worker_trial
  filedir = 'C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/resultsLunar/'
  if not os.path.exists(filedir):
      os.makedirs(filedir)

  filebase = filedir+"LunarLander-v2"+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  print("\n",filebase,"\n")
  controller = make_controller()

  num_params = controller.param_count
  print("size of model", num_params)

  cma = CMAES(num_params,
      sigma_init=sigma_init,
      popsize=population)
  es = cma

  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (5+num_params)*num_worker_trial #867
  RESULT_PACKET_SIZE = 2*num_worker_trial + 2 * num_episode * num_worker_trial # worker and job id for each worker + return list and timestep list for each worker    34

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions):
  result = []
  worker_num = 0
  for i in range(len(seeds)):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i]])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packet = np.asarray(packet)
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3:].astype(np.float)/PRECISION])
  return result

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, -1)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2:(2+num_worker_trial*num_episode)].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, (2+num_worker_trial*num_episode):].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def receive_packets_from_slaves(packet_list):
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)
  reward_list_total = np.zeros((population, num_episode*2)) #868
  check_results = np.ones(population, dtype=np.int)

  for i in range(1, num_worker):
    result_packet = packet_list[i-1]
    results = decode_result_packet(result_packet)
    for result in results:
      
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, :num_episode] = result[2]
      reward_list_total[idx, num_episode:] = result[3]
      check_results[idx] = 0
  return reward_list_total
	
def evaluate_batch(model_params, train_mode, max_len=-1):
    solutions = []
    for i in range(es.popsize):
      solutions.append(np.copy(model_params))
    seeds = np.arange(es.popsize)
    packet_list = encode_solution_packets(seeds, solutions)
    response_list_total = slave(packet_list)
    rewards_list = response_list_total[:, :(num_test_episode)].flatten()[:100] # get rewards

    return rewards_list

def master():
  for run_i in range(5):
    global test_env
    test_env = make_env(gym_env= "LunarLander-v2", dream_env=False,full_episode = False, load_model=True)

    start_time = int(time.time())
    print("training", env_name)
    print("population", es.popsize)
    print("num_worker", num_worker)
    print("num_worker_trial", num_worker_trial)
    sys.stdout.flush()

    seeder = Seeder(seed_start)

    filename = filebase+'.json'
    filename_hist = filebase+'.hist.json'
    filename_best = filebase+'.best.json'
  
    t = 0

    history = []


    filename_bestPerf = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstTransferbestLunarLander-v2"+".npy"
    filename_avg = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstTransferavgLunarLander-v2"+".npy"
    filename_worse = "C:/Users/benat/OneDrive/Dokumentuak/World Models CandNo215816/ResultsGraph"+"/"+"worstTransferLunarLander-v2"+".npy"


    best_performer = []
    avg_pop = []
    worst_performer = []

  

    for generation_i in range(20): #Generations
      solutions = es.ask()

      #if antithetic:
      #  seeds = seeder.next_batch(int(es.popsize/2))
      #  seeds = seeds+seeds
      #else:
      seeds = seeder.next_batch(es.popsize)
      packet_list = encode_solution_packets(seeds, solutions)
      packet_list = slave(packet_list)

      response_list_total = receive_packets_from_slaves(packet_list)

      reward_list_raw = response_list_total[:, :(num_episode)] # get rewards
      time_list_raw = response_list_total[:, (num_episode):]
      if batch_mode == 'min':
        reward_reduced = reward_list_raw.min(axis=1)
      elif batch_mode == 'mean':
        reward_reduced = reward_list_raw.mean(axis=1)
    # actual statistics from non reduced rewards
      mean_time_step = int(np.mean(time_list_raw)*100)/100. 
      max_time_step = int(np.max(time_list_raw)*100)/100.
      avg_reward = int(np.mean(reward_list_raw)*100)/100.
      std_reward = int(np.std(reward_list_raw)*100)/100. 
      es.tell(reward_reduced)

      es_solution= es.result()
      model_params = es_solution[0] # best historical solution
      reward = es_solution[1]
      curr_reward = es_solution[2]
      controller.set_model_params(np.array(model_params).round(4))

      r_max = int(np.max(reward_list_raw[0:-1])*100)/100.
      r_min = int(np.min(reward_list_raw[0:-1])*100)/100.

      curr_time = int(time.time()) - start_time

      h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

      best_performer.append(r_max)
      avg_pop.append(avg_reward)
      worst_performer.append(r_min)

      #if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
      #else:
      #  max_len = -1

      history.append(h)

      with open(filename, 'wt') as out:
        res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

      with open(filename_hist, 'wt') as out:
        res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

      with open(filename_best, 'wt') as out:
        res = json.dump([np.array(es.best_param()).round(4).tolist()], out, sort_keys=False, indent=0, separators=(',', ':')) 

      print(env_name, h)

      if (t == 1):
        best_reward_eval = avg_reward

      # increment generation
      t += 1
      np.save(filename_bestPerf+str(run_i), np.asarray(best_performer))
      np.save(filename_avg+str(run_i), np.asarray(avg_pop))
      np.save(filename_worse+str(run_i), np.asarray(worst_performer))

def slave(packet_list):
  global env, test_env
  result_packet = []
  env = make_env(gym_env="LunarLander-v2", dream_env=False,full_episode = False, load_model=True) 
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    solutions = decode_solution_packet(packet)
    results = []

    for solution in solutions:
      worker_id, jobidx, seed, weights = solution

      worker_id = int(worker_id)


      jobidx = int(jobidx)
      seed = int(seed)
      fitness, timesteps = worker(weights,seed)
      results.append([worker_id, jobidx, fitness, timesteps])
      result_packet.append(encode_result_packet(results))
  return result_packet

def worker(weights,seed):
  controller.set_model_params(weights)
  reward_list, t_list = simulateLunar(controller, env,
        render_mode=True, num_episode=num_test_episode, seed=-1)
  return reward_list, t_list

def encode_result_packet(results):
  r = np.reshape(np.array(results), [-1,])
  r = np.concatenate([np.array(A).flatten() for A in r], axis=0)
  eval_packet_size = 2*num_worker_trial + 2 * num_test_episode * num_worker_trial # not the same size for training
  r[2:] *= PRECISION
  if r.size == eval_packet_size:
      r = np.concatenate([r, np.zeros(RESULT_PACKET_SIZE - eval_packet_size)-1.0], axis=0)
  return r.astype(np.int32)
    
def main():
  global optimizer, num_episode, num_test_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, cap_time_mode, env_name, exp_name, batch_mode, config_args
  optimizer = 'cma'
  num_episode = 2 
  num_test_episode = 2
  eval_steps = 10
  num_worker = 32 
  num_worker_trial = 1
  antithetic = False
  retrain_mode = False
  cap_time_mode= False
  seed_start = 0
  env_name = 'Lunar Lander'
  exp_name = 'LunarLander-v2'
  batch_mode = 'mean'

  initialize_settings() 
  master()

  
if __name__ == "__main__":
  main()