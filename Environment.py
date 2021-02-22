import math
import numpy

from Coordinate import Coordinate

class Environment:

	def __init__(self, number_of_states=0, obstacle_rate=0):
		
		self.number_of_states = int(math.pow(int(math.sqrt(number_of_states)), 2))
		self.number_of_rows = int(math.sqrt(number_of_states))
		self.obstacle_rate = obstacle_rate
		self.reward_matrix = numpy.zeros((self.number_of_states, self.number_of_states))
		self.obstacles = list()       # List of coordinates
		self.obstacle_states = list() # List of obstacle states indexes
		self.goal_coordinate = Coordinate(0, 0)
		self.initial_state = Coordinate(0, 0)
		self.start_index = 0
		self.goal_index = 0
		
		#Generate the environment whenever new instance's being created
		#self.generate()


	def generate(self):

		"""
			this method generates the environment 
			with the given set of parameters in __init__
		"""
		
		for i in range(int(self.number_of_states)):
			row_state = Coordinate( int(i / self.number_of_rows) , i % self.number_of_rows)
			for j in range(int(self.number_of_states)):
				col_state = Coordinate( int(j / self.number_of_rows), j % self.number_of_rows)
				#Forbiding the passage from a state to NW(north west),SW,NE,SE
				if   row_state._get_col() == col_state._get_col() and row_state._get_row() - 1 == col_state._get_row():
					self.reward_matrix[(i, j)] = -1
				elif row_state._get_col() == col_state._get_col() and row_state._get_row() + 1 == col_state._get_row():
					self.reward_matrix[(i, j)] = -1
				elif row_state._get_col() - 1 == col_state._get_col() and row_state._get_row() == col_state._get_row():
					self.reward_matrix[(i, j)] = -1
				elif row_state._get_col() + 1 == col_state._get_col() and row_state._get_row() == col_state._get_row():
					self.reward_matrix[(i, j)] = -1
				else:
					self.reward_matrix[(i, j)] = -math.inf

		#Choosing a random goal state
		self.goal_coordinate = Coordinate.generate_random(math.sqrt(self.number_of_states))

		#Adding random obstacles to the environment
		for i in range(int(self.number_of_states * self.obstacle_rate)):	
			while True:
				obstacle = Coordinate.generate_random(math.sqrt(self.number_of_states))
				if not obstacle.equals(self.goal_coordinate) and obstacle not in self.obstacles:
					#Adding obstacle to the list of 
					#Obstacle coordinates
					#this list is represented as tuples
					# (row, col)
					tpl = obstacle.coo_tuple()
					self.obstacles.append(tpl)
					self.obstacle_states.append(tpl[0]*math.sqrt(self.number_of_states) + tpl[1])
					break
					
		#Choosing a random initial state
		while True:
			self.initial_state = Coordinate.generate_random(math.sqrt(self.number_of_states))
			
			if not self.initial_state.equals(self.goal_coordinate) :
				if self.initial_state.coo_tuple() not in self.obstacles :
					if math.fabs(self.initial_state.col - self.goal_coordinate.col) >= self.number_of_rows/2 :
						if math.fabs(self.initial_state.row - self.goal_coordinate.row) >= self.number_of_rows/2 :
							break

		# Calculate the goal state index 
		self.goal_index = self.goal_coordinate.coo_tuple()[0] * self.number_of_rows + self.goal_coordinate.coo_tuple()[1]
		
		# Start state index 
		self.start_index = self.initial_state.coo_tuple()[0] * self.number_of_rows + self.initial_state.coo_tuple()[1]
		
		# Looping when reached the goal state
		self.reward_matrix[(self.goal_index, self.goal_index)] = 1000

		# State in the west of the goal state ( white square)
		if self.goal_coordinate.col - 1 >= 0:
			west_state = self.goal_coordinate.row * self.number_of_rows + self.goal_coordinate.col -1
			self.reward_matrix[(west_state, self.goal_index)] = 1000
		
		# State in the east of the goal state(white square)
		if self.goal_coordinate.col + 1 < self.number_of_rows :
			east_state = self.goal_coordinate.row * self.number_of_rows + self.goal_coordinate.col + 1
			self.reward_matrix[(east_state, self.goal_index)] = 1000
		
		# State in the north of the goal state
		if self.goal_coordinate.row - 1 >= 0 :
			north_state = (self.goal_coordinate.row -1) * self.number_of_rows + self.goal_coordinate.col
			self.reward_matrix[(north_state, self.goal_index)] = 1000

		# State in the south of the goal state
		if self.goal_coordinate.row + 1 < self.number_of_rows :
			south_state = (self.goal_coordinate.row +1) * self.number_of_rows + self.goal_coordinate.col
			self.reward_matrix[(south_state, self.goal_index)] = 1000

		# Updating the reward matrix for obstacles
		for obstacle in self.obstacles :
			obstacle_index = obstacle[0] * self.number_of_rows + obstacle[1]
			# Check W, E, S, N of the obstacles
			# West
			if obstacle[1] - 1 >= 0 :
				west_index = Coordinate.coo_index(obstacle[0],obstacle[1]-1,self.number_of_rows)	
				self.reward_matrix[(west_index, obstacle_index)] = -math.inf
			# North
			if obstacle[0] - 1 >= 0 :
				north_index = Coordinate.coo_index(obstacle[0]-1,obstacle[1],self.number_of_rows)
				self.reward_matrix[(north_index, obstacle_index)] = -math.inf
			# East
			if obstacle[1] + 1 < self.number_of_rows :
				east_index = Coordinate.coo_index(obstacle[0],obstacle[1]+1,self.number_of_rows)
				self.reward_matrix[(east_index, obstacle_index)] = -math.inf
			# South
			if obstacle[0] + 1 < self.number_of_rows :
				south_index = Coordinate.coo_index(obstacle[0]+1,obstacle[1],self.number_of_rows)
				self.reward_matrix[(south_index, obstacle_index)] = -math.inf


	def draw_env(self):

		""" 
			Draws the environment after generating it  
		"""
		
		plot_matrix = numpy.zeros((self.number_of_rows, self.number_of_rows), dtype=numpy.int)
		for i in range(self.number_of_rows):
			for j in range(self.number_of_rows) :
				if Coordinate(i, j).equals(self.goal_coordinate):
					plot_matrix[(i, j)] = 6 
				elif Coordinate(i,j).equals(self.initial_state) :
					plot_matrix[(i, j)] = 5 
				elif (i,j) in self.obstacles :
					plot_matrix[(i, j)] = 1
				else :
					plot_matrix[(i, j)] = 0
		print(str(plot_matrix))
				

	
##############################################################################
###################            debugging               #######################
##############################################################################

if __name__ == "__main__":
	
	env = Environment(10, 0.1)
	env.generate()
	#print(env.reward_matrix)
	env.draw_env()
