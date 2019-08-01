#### feed in the obs seq line by line
#### quantity becomes all one
#### feed with all obs seq

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from hmm_class import hmm
from sklearn.preprocessing import normalize
import time
import matplotlib.pyplot as plt

# load the obs and split into k fold.
def split_load_data(filename, k_splits):
	obs_seq = np.loadtxt(filename, dtype = int)
	kf = KFold(n_splits = k_splits, shuffle = False)
	for train_index, test_index in kf.split(obs_seq):
		obs_train, obs_test = obs_seq[train_index], obs_seq[test_index]

	return obs_train, obs_test

def load_data(filename):
	obs_seq = np.loadtxt(filename, dtype = int)
	return obs_seq

# generate random states for the len of data
def sts_seq_generate(N, size_data, len_obs): # N states
	sts_seq = np.zeros((size_data, len_obs), dtype = int)
	for i in range(size_data):
		for j in range(len_obs):
			sts_seq[i][j] = random.randint(0,N-1)
	return sts_seq

 # generate emission probability randomly
 # return as matrix
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

# useful parameter for later use
def param_generate(n, obs_seq):
	size_data = len(obs_seq) # cal the line of obs 1000
	len_obs = len(obs_seq[0]) # cal the len of each obs. only works for the same length
	sts_seq = sts_seq_generate(n, size_data, len_obs)

	return size_data, len_obs, sts_seq

# output all the std file for this project
def outfile(filename,N = None, ep = None, tp = None, hidden_sts = None, distribution = None):
	f = open(filename, "w+")
	if N:
		f.write(str(N))
		f.write("\n")
	if np.any(ep):
		[n, m] = np.shape(ep)
		for i in range(n*m):
			f.write(str(ep.item(i)))
			if i % m == m-1:
				f.write("\n")
			else:
				f.write(",")
		for j in range(n*n):
			f.write(str(tp.item(j)))
			if j % n == n-1:
				f.write("\n")
			else:
				f.write(",")
	if hidden_sts:
		size_data = len(hidden_sts)
		len_seq = len(hidden_sts[0])
		for i in range(size_data):
			for j in range(len_seq):
				f.write(str(hidden_sts[i][j]))
				if j % len_seq == len_seq-1:
					f.write("\n")
				else:
					f.write(",")
	if distribution:
		size_data = len(distribution)
		len_seq = len(distribution[0])
		for i in range(size_data):
			for j in range(len_seq):
				f.write(str(distribution[i][j]))
				if j % len_seq == len_seq-1:
					f.write("\n")
				else:
					f.write(",")
	f.close()

# compute the predicted output prob distribution
# input the hidden states list and unique states outlook
# return the distribution list
def predic_prob(hidden_sts, uniq_sts):
	distribution = []
	each_prob = []
	size_data = len(hidden_sts)
	len_seq = len(hidden_sts[0])
	dis_dict = dict.fromkeys(uniq_sts, 0)
	for i in range(size_data):
		for j in range(len_seq):			
			dis_dict[hidden_sts[i][j]] += 1
			if j % len_seq == len_seq-1: # change line
				each_prob = prob_cal(dis_dict)
				distribution.append(each_prob)
				dis_dict.clear()
				dis_dict = dict.fromkeys(uniq_sts, 0)
	return distribution

# probability computation from the dictionary
def prob_cal(dictionary):
	prob_lst = []
	length = len(dictionary)
	total = []
	for i in range(length):
		total.append(dictionary[i])
	total = sum(total)
	for i in range(length):
		prob_lst.append(dictionary[i]/total)
	return prob_lst

# Given a seq of output, predict the next output and the next state, and test it on the data.
def predict_next_sts(hidden_sts, tp):
	size_data = len(hidden_sts)
	len_seq = len(hidden_sts[0])
	next_sts = []
	for i in range(size_data):
		next_sts.append(np.argmax(tp[hidden_sts[i][len_seq-1],:]))
	for i in range(size_data):
		hidden_sts[i].append(next_sts[i])
	return hidden_sts


if __name__ == '__main__':

	n = 5 # number of states
	m = 4 # number of observation
	k = 5 # k fold
	num_iter = 1000 # number of iteration
	tolerance = 10**(-5)

	obs_seq = load_data('train534.dat')
	size_data, len_obs, sts_seq = param_generate(n, obs_seq)
	uniq_sts = list(np.unique(sts_seq)) # the function need to feed in a list of uniq states
	uniq_obs = list(np.unique(obs_seq))

	pi = pi_generate(n) # start prob
	em_prob = em_prob_generate(n, m) # generate uniform distribution em prob
	trans_prob = trans_prob_generate(n) # generate uniform distribution trans prob.
	model = hmm(uniq_sts, uniq_obs, pi, trans_prob, em_prob) # init the model 

	# Number of times, each corresponding item in ‘observation_list’ occurs
	quantities = np.ones(size_data)

	prob = model.log_prob(obs_seq, quantities)
	print("prob of seq with original param %f" %(prob))

	# run EM/ Baum_welch to train the data.
	# use Baum_welch to maximize the likelihood
	# get the transition matrix A and emission probability matrix B
	ep, tp, sp, prob_lst, iter_count, loss_lst = model.train_hmm(obs_seq, num_iter, quantities, tolerance)

	print("emission_prob\n", ep)
	# pd.DataFrame(model.em_prob, index = uniq_sts, columns = uniq_obs)
	print("pi\n", sp)
	print("transition\n", tp)

	prob = model.log_prob(obs_seq, quantities)
	print("prob of seq after %d iterations: %f" %(num_iter, prob))

	# use viterbi to compute the most likely sequence. Report the time it took.	
	tr_start_t = time.perf_counter()
	hidden_states = []
	for i in range(size_data):
		hidden_states.append(model.viterbi(obs_seq[i]))
	tr_end_t = time.perf_counter()
	print("time for get all the hidden states from training data:", tr_end_t - tr_start_t)

	# print('hidden states:\n', hidden_states)


	###### calculate the log likelihood of test set
	###### predict the output from test seq
	test_obs = load_data('test1_534.dat')	
	size_data_test, len_obs_test, test_sts_seq = param_generate(n, test_obs)
	test_quant = np.ones(size_data_test)
	test_prob = model.log_prob(test_obs, test_quant)
	print("The log likelihood of test set: %f" %(test_prob))

	##### output the hidden states of test set
	te_start_t = time.perf_counter()
	test_hidden_sts = []
	for i in range(size_data_test):
		test_hidden_sts.append(model.viterbi(test_obs[i]))
	te_end_t = time.perf_counter()
	print("time for get all the hidden states from test data:", te_end_t - te_start_t)
	test_hidden_sts = [list(map(int, lst)) for lst in test_hidden_sts] # cast the data to int
	# comput the next state for every sequnece. T=40 to 41
	test_hidden_sts = predict_next_sts(test_hidden_sts, tp)
	# print("test set hidden states:\n", test_hidden_sts)
	distribution = predic_prob(test_hidden_sts, uniq_sts)


	####### output file ##########
	outfile("modelpars.dat", N = n, ep = ep, tp = tp)
	outfile("loglik.dat", N = test_prob)
	outfile("viterbi.dat", hidden_sts = test_hidden_sts)
	outfile("predict.dat", distribution = distribution)

	####### plot ###########
	x = np.arange(0, iter_count)
	plt.figure()
	plt.plot(x, prob_lst, color = 'r')
	plt.xlabel('iteration times')
	plt.ylabel('log likelihood')
	plt.title('Learning curve')
	plt.show()

	plt.figure()
	plt.plot(x, loss_lst, color = 'b')
	plt.xlabel('iteration times')
	plt.ylabel('loss')
	plt.title('Loss from each iteration')
	plt.show()











	# ep_test, tp_test, sp_test = model.train(obs_test, 2, quantities_test)
	# run the baum-welch algo to obtain the A, B matrix and start prob.





# def em_prob_generate():
# 	return np.matrix('0.1 0.2 0.3 0.4; 0.2 0.3 0.1 0.4; 0.4 0.3 0.2 0.1; 0.2 0.1 0.4 0.3; 0.3 0.1 0.2 0.4')

# def trans_prob_generate():
# 	return np.matrix('0.2 0.1 0.3 0.2 0.2; 0.1 0.2 0.2 0.1 0.4; 0.3 0.1 0.1 0.2 0.3; 0.2 0.1 0.1 0.2 0.4; 0.3 0.3 0.2 0.1 0.1')

# def pi_generate():
# 	return np.matrix('0.1 0.2 0.3 0.1 0.3')