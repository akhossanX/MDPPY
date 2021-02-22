import math
import time

from Environment import *
from random import randint, sample



class Algorithms :
	
	"""
		Responsible of resolving the MDP problem using 3 different
		Algorithms:
		+ Value Iteration Algorithm
		+ Policy Iteration Algorithm
		+ Modified Value Iteration ( Accelerated VI )
	"""

	def value_iteration(cls, environment=None, threshold=0, discount=0) :

		"""
			Solves the MDP using value iteration algorithm given:
			+ Environment: model of the Markov Decision Process environment
			+ Threshold  : the parameter epsilon used by VI
			+ Discount   : the discount factor must be >0 and <1
		"""
		
		# How much time does the algorithm take to converge 
		execution_time = 0
		# How much iteration the algorithm takes to converge
		iterations = 0
		# Reward matrix
		reward_matrix = environment.reward_matrix
		# Policy
		policy = {}
		# Initializing the policy with -1 
		policy = {i: -1 for i in range(environment.number_of_states)}
		# Obstacles (represented by a list of indices)
		obstacles = environment.obstacle_states
		#Utility functions
		utility = []
		# Initializing the utility function with 0
		utility_update = [ 0 for i in range(environment.number_of_states) ]
		fuzziness = threshold * (1 - discount) / (2 * discount)
		start_time = time.clock() 
		while True:
			delta_max = 0
			iterations += 1
			utility = utility_update.copy()
			for p_state in range(len(reward_matrix)) :
				# p_state can't be an obstacle
				if p_state not in obstacles :
					utility_update[p_state] = -math.inf
					for s_state in range(len(reward_matrix)) :
						# Exploring successor states
						reward = reward_matrix[(p_state, s_state)] + discount * utility[s_state]
						# Update value Function Maximum
						if utility_update[p_state] < reward :
							utility_update[p_state] = reward
							# Adding s_state to policy
							policy[p_state] = s_state				
				else :
					utility_update[p_state] = -math.inf
			# Updating delta after each iteration
			for i in range(len(utility_update)) :
				diff = abs(utility_update[i]-utility[i])
				if diff > delta_max :
					delta_max = diff 
			if delta_max < fuzziness :
				break
		end_time = time.clock() 
		execution_time = end_time - start_time
		return AlgorithmResult(policy, utility, iterations, execution_time)

	value_iteration = classmethod(value_iteration)




	def accelerated_value_iteration(cls, environment=None, threshold=0, discount=0) :

		"""
			Solves the MDP using a variant of value iteration
			algorithm given:
			+ Environment: model of the MDP environment
			+ Threshold  : the parameter epsilon used by VI
			+ Discount   : the discount factor must be >=0 and <1
		"""
		
		# How much time does the algorithm take to converge 
		execution_time = 0
		# How much iteration the algorithm takes to converge
		iterations = 0
		# Reward matrix
		reward_matrix = environment.reward_matrix
		# Policy
		policy = {}
		# Initializing the policy with -1 
		policy = {i: -1 for i in range(environment.number_of_states)}
		# Obstacles (represented by a list of indexes)
		obstacles = environment.obstacle_states
		#Utility functions
		utility = []
		utility_update = []
		# Initializing the utility function with 0
		utility_update = [ 0 for i in range(environment.number_of_states) ]
		fuzziness = threshold * (1 - discount) / (2 * discount)

		start_time = time.clock() 
		while True:
			delta_max = 0
			iterations += 1
			utility = utility_update.copy()
			for p_state in range(len(reward_matrix)) :
				# p_state can't be an obstacle
				if p_state not in obstacles :
					utility_update[p_state] = - math.inf
					for s_state in range(len(reward_matrix)) :
						# s_state must be the dircet adjacent successor of p_state
						if reward_matrix[(p_state, s_state)] != -math.inf and s_state != p_state :
							# Applying Gauss-Seidel operator
							if s_state < p_state :
								reward = reward_matrix[(p_state, s_state)] + discount * utility_update[s_state]
							else:
								reward = reward_matrix[(p_state, s_state)] + discount * utility[s_state]
						# Update value Function Maximum
							if utility_update[p_state] < reward :
								utility_update[p_state] = reward
								# Adding s_state to policy
								policy[p_state] = s_state				
				else :
					utility_update[p_state] = -math.inf

			# Updating delta after each iteration
			for i in range(len(utility_update)) :
				diff = abs(utility_update[i]-utility[i])
				if diff > delta_max :
					delta_max = diff 
			if delta_max < fuzziness :
				break
		
		end_time = time.clock() 
		execution_time = end_time - start_time
		
		return AlgorithmResult(policy, utility, iterations, execution_time)

	accelerated_value_iteration = classmethod(accelerated_value_iteration)


	def policy_iteration(cls, env, discount) :

		"""
			Solves the MDP using a policy iteration
			algorithm given:
			+ Environment: model of the MDP environment
			+ Threshold  : the parameter epsilon used by VI
			+ Discount   : the discount factor must be >=0 and <1
		"""
	
		execution_time = 0
		reward_matrix = env.reward_matrix
		obstacles = env.obstacle_states
		no_change = False		
		policy = {}
		iterations = 0
		utility = [0 for i in range(len(reward_matrix))]
		# Initializing policy with random values
		for i in range(len(utility)):
			if i not in obstacles :
				policy[i] = Algorithms.random_state(reward_matrix, i) 
			else:	
				policy[i] = -1
		start_time = time.clock()

		while not no_change:
			no_change = True
			iterations += 1
			# Policy evaluation
			pol_ev = Algorithms.policy_evaluation(policy, utility, reward_matrix, discount) 
			utility = pol_ev.copy()
			# Selecting actions maximazing the utility
			for p_state in range(len(reward_matrix)) :
				max_utility = utility[p_state]
				if p_state not in obstacles:
					for s_state in range(len(reward_matrix)):
						reward = reward_matrix[(p_state, s_state)] + discount * utility[s_state]
						if reward > max_utility : 
							policy[p_state] = s_state
							max_utility = reward
							no_change = False
		end_time = time.clock()
		execution_time = end_time - start_time
		
		return AlgorithmResult(policy, utility, iterations, execution_time)
			
	policy_iteration = classmethod(policy_iteration)	

	def random_state(cls, rew_mat, current_state):

		"""
			Generates a random state index
		"""
		random_state  = -1
		next_states = []
		#We choose only adjacent states
		for j in range(len(rew_mat)):
			if rew_mat[(current_state, j)] != -math.inf :
				next_states.append(j)
		#Then we randomly select one of them
		if len(next_states) != 0 :
			random_state = sample(next_states, 1)[0]
		return random_state 
	random_state = classmethod(random_state)

	def policy_evaluation(cls, policy, utility, reward_matrix, discount) :

		"""
			Evaluates a given policy using iterative approach
		"""	
		new_utility = utility.copy()
		for i in range(20):
			for p_state in range(len(policy)):
				if policy[p_state] != -1 :
					if reward_matrix[(p_state, policy[p_state])] != -math.inf:
						new_utility[p_state] = reward_matrix[(p_state, policy[p_state])] + \
						discount * new_utility[policy[p_state]]
		return new_utility

	policy_evaluation = classmethod(policy_evaluation)





class AlgorithmResult:

	""" 
		This class represents the result of solving MDP with a given algorithm
	"""
	def __init__(self, policy, utility, iterations, execution_time):
		self.policy = policy
		self.utility = utility
		self.iterations = iterations
		self.execution_time = execution_time


if __name__ == "__main__":
	env = Environment(1500, 0.2)
	env.generate()
	res = Algorithms.policy_iteration(env, 0.9)
	print(res.iterations)
	print(res.execution_time)
	print(res.policy)
