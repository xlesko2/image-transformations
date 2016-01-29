import Image
import math
import threading

from transformations import t_rotate, t_resize, t_translate, t_detect_edges, t_grayscale, t_noise_hugo, t_filter_dante, t_blend_color, t_blend_image, t_blur, e_detect_objects, e_solve_maze

# COLORS
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)

class KL_Image:
	def __init__(self, source):
		return self._reset(source)
	
	def generate(self, size, color):
		self.image = Image.new('RGB', size, color)
		return None
	
	def _reset(self, source):
		try:
			self.image = Image.open(source)
		except IOError:
			print "The image doesn't (probably) exist."
			self.image = Image.new('RGB', (0,0), (0,0,0))
		
		self.size = self.image.size
		self.mode = self.image.mode
		self.source = source
		return None
	
	def save(self, target):
		self.image.save(target)
		return None
	
	def reset(self):
		return self._reset(self.source)
	
	def show(self):
		return self.image.show()
	
	def rotate(self, angle, center, bg):
		self.image = t_rotate(self.image, angle, center, bg)
		self.size = self.image.size
		return None
	
	def resize(self, ratio_w, ratio_h):
		self.image = t_resize(self.image, ratio_w, ratio_h)
		self.size = self.image.size
		return None
	
	def translate(self, vector, loop = True):
		self.image = t_translate(self.image, vector, loop)
		return None
	
	def detect_edges(self, resolution, sampling, threshold, edge_color):
		self.image = t_detect_edges(self.image, resolution, sampling, threshold, edge_color)
		return None
	
	def grayscale(self, mode):
		self.image = t_grayscale(self.image, mode)
		return None
	
	def hugo_noise(self):
		self.image = t_noise_hugo(self.image)
		return None
	
	def dante_filter(self, threshold, background, foreground):
		self.image = t_filter_dante(self.image, threshold, background, foreground)
		return None
	
	def blend_color(self, blend_color, blend_level):
		self.image = t_blend_color(self.image, blend_color, blend_level)
		return None
	
	def blend_image(self, blend_image, blend_level):
		self.image = t_blend_image(self.image, blend_image, blend_level)
		return None
	
	def blur(self, range_x, range_y):
		self.image = t_blur(self.image, range_x, range_y)
		return None
	
	def detect_objects(self, threshold):
		self.image = e_detect_objects(self.image, threshold)
		return None
	
	def solve_maze(self, space_color, wall_color, start_color, end_color, path_color):
		self.image = e_solve_maze(self.image, space_color, wall_color, start_color, end_color, path_color)
		return None


a = KL_Image('source.jpg')

# Maze pathfinding
# 	image source: http://holoweb.net/~liam/pictures/drawings/whitemaze-1390x1090.jpg
'''
a = KL_Image('whitemaze.jpg')
a.image = a.image.convert('RGB')
b = a.image.load()
a.solve_maze(WHITE, BLACK, GREEN, RED, MAGENTA)
a.show()
'''
