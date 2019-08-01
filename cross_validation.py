# this is to use cross validation to obtain the number of states
import numpy as np
from sklearn.model_selection import KFold
from hmm_class import hmm
import random
from sklearn.preprocessing import normalize

# load the obs and split into k fold.
def split_load_data(filename, k_splits):
	obs_seq = np.loadtxt(filename, dtype = int)
	kf = KFold(n_splits = k_splits, shuffle = False)
	for train_index, test_index in kf.split(obs_seq):
		obs_train, obs_test = obs_seq[train_index], obs_seq[test_index]

	return obs_train, obs_test

def sts_seq_generate(N, size_data, len_obs): # N states
	sts_seq = np.zeros((size_data, len_obs))
	for i in range(size_data):
		for j in range(len_obs):
			sts_seq[i][j] = random.randint(0,N-1)
	return sts_seq

def param_generate(n, obs_seq):
	size_data = len(obs_seq) # cal the line of obs 1000
	len_obs = len(obs_seq[0]) # cal the len of each obs. only works for the same length
	sts_seq = sts_seq_generate(n, size_data, len_obs)

	return size_data, len_obs, sts_seq

def em_prob_generate(n, m): # n:# states, m: # obs
	em_prob = np.zeros((n,m))
	for i in range(n):
		for j in range(m):
			em_prob[i][j] = np.random.uniform(0,1)
	em_prob = normalize(em_prob, axis = 1, norm = 'l1')
	return np.asmatrix(em_prob)

def trans_prob_generate(n): # n:# states
	trans_prob = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			trans_prob[i][j] = np.random.uniform(0,1)
	trans_prob = normalize(trans_prob, axis = 1, norm = 'l1')
	return np.asmatrix(trans_prob)

def pi_generate(n):
	pi = np.zeros(n)
	for i in range(n):
		pi[i] = np.random.uniform(0,1)
	pi = normalize([pi], axis = 1, norm = 'l1')
	return np.asmatrix(pi)

def predict_obs(first_sts,ep, tp):
	predicted_obs = []
	next_sts = np.argmax(tp[int(first_sts),:]) # from the first to compute 2nd states
	for _ in range(len_obs_test):
		# from the next sts to find out the argmax of the index
		# so that we know the most emission output		
		out = np.argmax(ep[next_sts,:])
		# print("obs index", out)
		predicted_obs.append(uniq_obs[out]) # from the output list to take out the obs
		next_sts = np.argmax(tp[next_sts,:]) # update the next sts
		# print("next states: ", next_sts)
	return predicted_obs

def loss_count(hidden_sts, predicted_obs):
	loss = 0
	hidden_sts = [int(i) for i in hidden_sts]
	for i in range(len(hidden_sts)):
		if hidden_sts[i] != predicted_obs[i]:
			loss += 1
	return loss

if __name__ == '__main__':
	n = 6 # number of states
	m = 4 # number of obs
	k = 5 # number of fold
	num_iter = 1000
	tolerance = 10**(-4)

	obs_train, obs_test = split_load_data('train534.dat', k)

	# train data
	size_train_data, len_obs_train, sts_seq_train = param_generate(n, obs_train)

	# test data
	size_test_data, len_obs_test, sts_seq_test = param_generate(n, obs_test)

	# uniq seq of sts and obs
	uniq_sts = list(np.unique(sts_seq_train)) # the function need to feed in a list of uniq states
	uniq_obs = list(np.unique(obs_train))
	quantities = np.ones(size_test_data)

	# prob param
	pi = pi_generate(n)
	trans_prob = trans_prob_generate(n)
	em_prob = em_prob_generate(n, m)

	model = hmm(uniq_sts, uniq_obs, pi, trans_prob, em_prob) # init the model
	ep, tp, sp, prob_lst, iter_count, loss_lst = model.train_hmm(obs_test, num_iter, quantities, tolerance)

	hidden_sts = []
	for i in range(size_train_data):
		hidden_sts.append(model.viterbi(obs_train[i]))

	# print("hidden states for training data:\n", hidden_sts)
	# print(len(hidden_sts))
	print("em_prob\n", ep)
	print("trans_prob\n", trans_prob)

	# predicted output from the training data
	first_index = []
	for i in range(len(hidden_sts)):
		first_index.append(hidden_sts[i][0])

	predicted_obs = []
	for i in range(len(hidden_sts)):
		predicted_obs.append(predict_obs(first_index[i], ep, tp))
	# print("predicted obs", predicted_obs)

	loss = []
	for i in range(len(hidden_sts)):
		loss.append(loss_count(hidden_sts[i], predicted_obs[i]))

	print("total loss: ", sum(loss))
