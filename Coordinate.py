from random import randint

class Coordinate:
	
	"""
		Represents the coordinates (row, column) of 
		each state in the environment
	"""
	
	def __init__(self, row, col):
		
		self._row = row
		self._col = col

	def _get_row(self):
		return self._row

	def _set_row(self, row):
		self._row = row

	def _get_col(self):
		return self._col

	def _set_col(self, col):
		self._col = col

	row = property(_get_row, _set_row)
	col = property(_get_col, _set_col)

	def equals(self, Object):
		"""
			Returns True if self and object are the same 
			environment state
		"""

		return (type(self) is type(Object)) and  (self._get_row() == Object._get_row()) and (self._get_col() == Object._get_col()) 

	def generate_random(cls, Max=0):	
		"""
			Generates a random state coordinate
		"""
		return Coordinate(randint(0, Max-1), randint(0, Max-1))	
	generate_random = classmethod(generate_random)

	# Method representing coordinates as a tuple
	def coo_tuple(self):
		"""
			Converts Coordinate object to tuple
		"""
		return (self.row, self.col)

	# Calculates the index of a given coordinate
	def coo_index(cls, row, col, number_of_rows):
		"""
			 Calculates the index of a state given its row and column 
		"""
		return row * number_of_rows + col
	coo_index = classmethod(coo_index)
