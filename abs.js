# based on example on https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/egt/examples/alpharank_example.py
# and pyten/testImgRecovery.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel 
import numpy as np
from pyten.tools import tenerror
from pyten.tenclass import tensor
from pyten.method import *
import scipy.stats as stats
import tensorly as tl
from tensorly.decomposition import tucker,CP, parafac
import csv
from numpy.linalg import matrix_rank

NUM_STRATEGIES = 1000
NUM_PLAYERS = 4
HUMAN_EST_PAYOFF_TENSOR_CP_RANK = 200


def get_game_fictitious_play_payoff_data(game, num_players, game_settings):
  """Returns the kuhn poker data for the number of players specified."""
  game = pyspiel.load_game(game, game_settings)
  xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
  for _ in range(999):
  for _ in range(NUM_STRATEGIES - 1):
    xfp_solver.iteration()

  meta_games = xfp_solver.get_empirical_metagame(100, seed=27)
  # Metagame utility matrices for each player
  payoff_tables = []
  for i in range(num_players):
    payoff_tables.append(meta_games[i])
  return payoff_tables

def main(unused_arg):
  # Construct meta-game payoff tables
  num_players = 4
  payoff_tables = get_game_fictitious_play_payoff_data('kuhn_poker', num_players, {'players': num_players})
  payoff_tables = get_game_fictitious_play_payoff_data('kuhn_poker', NUM_PLAYERS, {'players': NUM_PLAYERS})

  with open('payoff_open_spiel_kuhn_poker_seed27_data.npy', 'wb') as f:
    np.save(f, np.array(payoff_tables))
  f = open('synthetic_experiment_result.csv', 'w')
  writer = csv.writer(f)
  ###### Run AlphaRank with ground truth #####
  payoffs_are_hpt_format_original = utils.check_payoffs_are_hpt(payoff_tables)
  strat_labels_original = utils.get_strat_profile_labels(payoff_tables,
                                                payoffs_are_hpt_format_original)
  rhos, rho_m, pi, _, _ = alpharank.compute(payoff_tables, alpha=1e2)
  # Report & plot results
  utils.print_rankings_table(payoff_tables, pi, strat_labels_original, num_top_strats_to_print=10)

  for data_keep_rate in np.arange(0.3,0.99,0.01):
  for data_keep_rate in np.arange(0.3,0.99,0.1):
    ####### setup tensor completion ##########
    payoff_shape = payoff_tables[0].shape
    omegas = []
    rng = np.random.default_rng(int(data_keep_rate*100))
    omega = (rng.random(payoff_shape) <= data_keep_rate) * 1
    print("np.count_nonzero(omega==0): ", np.count_nonzero(omega==0))
    omegas = [omega] * NUM_PLAYERS
    payoff_missing_tensors = []
    for i in range(len(payoff_tables)):
      rng = np.random.default_rng(i*7)
      omega = (rng.random(payoff_shape) <= data_keep_rate) * 1
      print("np.count_nonzero(omega==0): ", np.count_nonzero(omega==0))
    for i in range(len(payoff_tables)):      
      X1 = tensor.Tensor(payoff_tables[i].copy())
      X1.data[omega == 0] = 0
      print("np.count_nonzero(X1.data==0): ", np.count_nonzero(X1.data==0))
      payoff_missing_tensors.append(X1)
      omegas.append(omega)
      # omegas.append(omega)

    ###### tensor completion #########
    completed_payoff_tensors = []
    r = 200 # human estimation
    for i in range(len(payoff_missing_tensors)):
      tncp1 = TNCP(payoff_missing_tensors[i], omegas[i], rank=r, tol=1e-15, max_iter=5000, printitn=0) # 
      tncp1 = TNCP(payoff_missing_tensors[i], omegas[i], rank=HUMAN_EST_PAYOFF_TENSOR_CP_RANK, tol=1e-15, max_iter=5000, printitn=0) # 
      tncp1.run()
      recoerved_result = tncp1.X
      completed_payoff_tensors.append(recoerved_result)
      [Err1, ReErr11, ReErr21] = tenerror(recoerved_result, payoff_tables[i], omegas[i])
      print ('\n', 'The frobenius norm of error between recovered and original tensor are:', Err1, ReErr11)
    completed_payoff_tables = []
    for cpt in completed_payoff_tensors:
      completed_payoff_tables.append(cpt.data)
    
    ###### alpharank with completed tensors ##########
    try:
      payoffs_are_hpt_format = utils.check_payoffs_are_hpt(completed_payoff_tables)
      strat_labels = utils.get_strat_profile_labels(completed_payoff_tables, payoffs_are_hpt_format)
      rhos_est, rho_m_est, pi_est, _, _ = alpharank.compute(completed_payoff_tables, alpha=1e2)
      # print(pi_est)
      utils.print_rankings_table(completed_payoff_tables, pi_est, strat_labels, num_top_strats_to_print=10)
    except ValueError as e:
      print(e)
      print("ValueError!!!", e)
      continue

    ##### alpharank with original payoff #################
    # print metrics to measure alpharank diffs
    print("max pi error: ",np.max(np.abs(pi_est - pi)))
    print("pi RMSE:", np.sqrt(np.mean((pi_est-pi)**2)))
    tau, p_value = stats.kendalltau(pi, pi_est)
    print("tau: ", tau)
    # print('p_value', p_value)
    writer.writerow([data_keep_rate, np.max(np.abs(pi_est - pi)), np.sqrt(np.mean((pi_est-pi)**2)), tau])
    print([data_keep_rate, np.max(np.abs(pi_est - pi)), np.sqrt(np.mean((pi_est-pi)**2)), tau])
if __name__ == '__main__':
  app.run(main)
  23  
pyten/method/TNCP.py
@@ -1,6 +1,7 @@
import numpy as np
import pyten.tenclass
import pyten.tools
import pyten.tenclass as tenclass
import pyten.tools as tools
import pyten.tenclass.tenmat as tenmat


class TNCP(object):
@@ -23,19 +24,19 @@ def __init__(self, obser, omega=None, rank=20, tol=1e-5, max_iter=500, alpha=Non
                 printitn=500):
        if not obser:
            raise ValueError("TNCP: observed Tensor cannot be empty!")
        elif type(obser) != pyten.tenclass.Tensor and type(obser) != np.ndarray:
            raise ValueError("TNCP: cannot recognize the format of observed Tensor!")
        elif type(obser) != tenclass.tensor.Tensor and type(obser) != np.ndarray:
            raise ValueError(f"TNCP: cannot recognize the format of observed Tensor! {type(obser)}")
        elif type(obser) == np.ndarray:
            self.T = pyten.tenclass.Tensor(obser)
            self.T = tenclass.Tensor(obser)
        else:
            self.T = obser

        if omega is None:
            self.omega = self.T.data * 0 + 1
        if type(omega) != pyten.tenclass.Tensor and type(omega) != np.ndarray:
        if type(omega) != tenclass.tensor.Tensor and type(omega) != np.ndarray:
            raise ValueError("TNCP: cannot recognize the format of indicator Tensor!")
        elif type(omega) == np.ndarray:
            self.Omega = pyten.tenclass.Tensor(omega)
            self.Omega = tenclass.tensor.Tensor(omega)
        else:
            self.Omega = omega

@@ -79,9 +80,9 @@ def initializeLatentMatrices(self):
        self.U = [np.random.rand(self.shape[i], self.rank) for i in range(self.ndims)]
        self.Y = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.Z = [np.zeros((self.shape[i], self.rank)) for i in range(self.ndims)]
        self.II = pyten.tools.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        self.II = tools.tendiag(np.ones(self.rank), [self.rank for i in range(self.ndims)])
        self.X = self.T.data + (1 - self.Omega.data) * (self.T.norm() / self.T.size())
        self.X = pyten.tenclass.Tensor(self.X)
        self.X = tenclass.tensor.Tensor(self.X)
        self.X_pre = self.X.copy()

    def run(self):
@@ -116,13 +117,13 @@ def run(self):
                    if j == i:
                        continue
                    midT = midT.ttm(self.U[j], j + 1)
                unfoldD_temp = pyten.tenclass.Tenmat(midT, i + 1)
                unfoldD_temp = tenmat.Tenmat(midT, i + 1)

                temp_Z = self.eta * self.Z[i] + self.Y[i]
                temp_B = np.dot(unfoldD_temp.data, unfoldD_temp.data.T)
                temp_B += self.eta * np.identity(self.rank)
                temp_B += 0.00001 * np.identity(self.rank)
                temp_C = pyten.tenclass.Tenmat(self.X, i + 1)
                temp_C = tenmat.Tenmat(self.X, i + 1)
                temp_D = np.dot(temp_C.data, unfoldD_temp.data.T)
                self.U[i] = np.dot((temp_D + temp_Z), np.linalg.inv(temp_B))
