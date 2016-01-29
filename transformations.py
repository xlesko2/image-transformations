# TRANSFORMATIONS
# This file holds particular image transformation functions for KL_Image class.

### Imports

import Image
from random import randrange, random
from math import sin, cos, pi, sqrt, floor

### Functions

def t_rotate(source, angle, center, background):
	'''
	ROTATION
	Works by interpreting the image as an array of points in E2 and rotating them around an arbitrarily defined center.
	Main function handling rotation.
	Params:
		source	-- Image object to process
		angle	-- angle of rotation (in radians)
		center	-- center of rotation (can be None, which will default to image center)
		background	-- color with which to cover pixels that are left empty by rotation
	'''
	
	# Aux. function for averaging the color of neighbouring pixels to cover 'holes' left by rotating the image.
	def average_color(pixelmap, pixel):
		averaged_pixels = list()
		for x in range(-1, 2):
			for y in range(-1, 2):
				if not ((x == 0) and (y == 0)):
					try:
						averaged_pixels.append(pixelmap[pixel[0] + x, pixel[1] + y])
					except IndexError: # pixel can be located on the edge of the image; doesn't matter
						pass
		return tuple([sum([j[i] for j in averaged_pixels])/len(averaged_pixels) for i in range(3)])
	
	# Aux. function for rotating a single pixel around the center.
	def rotate_pixel(position, angle, center):
		vector = (position[0] - center[0], position[1] - center[1])
		rotated_vector = (cos(angle) * vector[0] - sin(angle) * vector[1], sin(angle) * vector[0] + cos(angle) * vector[1])
		return (int(round(center[0] + rotated_vector[0])), int(round(center[1] + rotated_vector[1])))
	
	source_pixelmap = source.load()
	
	if center == None:
		center = tuple([i/2 for i in source.size])
	
	result_matrix = dict()
	for x0 in range(source.size[0]):
		for y0 in range(source.size[1]):
			result_matrix[(x0,y0)] = rotate_pixel((x0, y0), angle, center)
	
	width_max = max(result_matrix.values(), key = lambda pixel: pixel[0])[0]
	width_min = min(result_matrix.values(), key = lambda pixel: pixel[0])[0]
	width_new = width_max - width_min + 1
	height_max = max(result_matrix.values(), key = lambda pixel: pixel[1])[1]
	height_min = min(result_matrix.values(), key = lambda pixel: pixel[1])[1]
	height_new = height_max - height_min + 1
	
	result = Image.new("RGB", (width_new, height_new), background)
	result_pixelmap = result.load()
	covered_pixels = [[0 for j in range(result.size[1])] for i in range(result.size[0])]
	for pixel in result_matrix:
		result.putpixel((result_matrix[pixel][0] - width_min, result_matrix[pixel][1] - height_min), source_pixelmap[pixel])
		covered_pixels[result_matrix[pixel][0] - width_min][result_matrix[pixel][1] - height_min] = 1
	
	for x1 in range(result.size[0]):
		for y1 in range(result.size[1]):
			if covered_pixels[x1][y1] == 0:
				result_pixelmap[x1,y1] = average_color(result_pixelmap, (x1,y1))
	
	return result


def t_resize(source, ratio_w, ratio_h):
	'''
	RESIZE
	Changes image size by given ratio (may be different for width and height). Works (roughly) by nearest neighbor interpolation.
	Params:
		source	-- Image object to process
		ratio_w	-- width resize ratio (float)
		ratio_h	-- height resize ratio (float)
	'''
	
	source_pixelmap = source.load()
	new_size = (int(round(source.size[0] * ratio_w)), int(round(source.size[1] * ratio_h)))
	result = Image.new(source.mode, new_size, (0,0,0))
	result_pixelmap = result.load()
	
	for x in range(new_size[0]):
		for y in range(new_size[1]):
			result_pixelmap[x,y] = source_pixelmap[int(floor(x / ratio_w)), int(floor(y / ratio_h))]
	return result


def t_translate(source, vector, loop):
	'''
	TRANSLATION
	Function for moving all pixels of an image by a given vector.
	Params:
		source	-- Image object to process
		vector	-- 2D translation vector
		loop	-- toggle looping the pixels if they move out of screen (boolean)
	'''
	
	# Auxiliary function for moving a single pixel (incl. looping if it gets out of the image).
	def translate_pixel(pixel, vector, size, loop):
		result_position = (pixel[0] + vector[0], pixel[1] + vector[1])
		if loop:
			result_position = tuple([result_position[i] % size[i] for i in xrange(2)])
		return result_position
	
	source_pixelmap = source.load()
	
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			translated_pixel = translate_pixel((x,y), vector, source.size, loop)
			try:
				result_pixelmap[translated_pixel] = source_pixelmap[x,y]
			except IndexError:
				pass
	return result


def t_detect_edges(source, resolution, sampling, threshold, edge_color):
	'''
	EDGE DETECTION
	Works by randomly selecting a large number of vectors and comparing the color difference in their end points.
	Params:
		source		-- Image object to be processed
		resolution	-- length of edge-testing vectors
		sampling	-- number of samples (test vectors) to execute
		threshold	-- minimum level of color difference to consider an edge
		edge_color	-- color to use for edge highlighting
	'''
	
	# Aux. function for rotating a 2D vector.
	def rotate_vector(vector, angle):
		return (int(round(cos(angle) * vector[0] - sin(angle) * vector[1])), int(round(sin(angle) * vector[0] + cos(angle) * vector[1])))

	# Aux. function calculating the color difference. Uses unit cube as its reference.
	def calculate_difference(color1, color2):
		return sqrt((color2[0] - color1[0])**2 + (color2[1] - color1[1])**2 + (color2[2] - color1[2])**2)/sqrt(3 * 255**2)

	# Aux. function for finding center of a vector (specifying the pixel where an edge was found).
	def find_center(point1, point2):
	    return ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)
	
	source_pixelmap = source.load()
	result = Image.new("RGB", source.size, (0, 0, 0))
	result_pixelmap = result.load()
	
	for i in range(sampling):
		point1 = randrange(source.size[0]), randrange(source.size[1])
		point2 = (-1, -1)
		while not ((point2[0] in xrange(source.size[0])) and (point2[1] in xrange(source.size[1]))): # and (result.getpixel(find_center(point1, point2)) <> EDGE_COLOR)
			rotated_vector = rotate_vector((resolution, 0), 2 * pi * random())
			point2 = (point1[0] + rotated_vector[0], point1[1] + rotated_vector[1])
		color_difference = calculate_difference(source_pixelmap[point1], source_pixelmap[point2])
		if color_difference > threshold:
			point3 = find_center(point1, point2)
			result_pixelmap[point3] = edge_color
	return result


def t_grayscale(source, mode):
	'''
	CONVERT TO GRAYSCALE
	Makes image grayscale by only preserving a single color channel (or average of all three).
	Params:
		source	-- Image object to process
		mode	-- 0 - preserve R channel, 1 - preserve G channel, 2 - preserve B channel, 3 - preserve average or RGB channels
	'''
	
	source_pixelmap = source.load()
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			pixel = source_pixelmap[x,y]
			if mode == 3:
				average = int(round(float(sum(pixel))/3))
				result_pixelmap[x,y] = tuple([average]*3)
			else:
				result_pixelmap[x,y] = tuple([pixel[mode]] * 3)
	return result


def t_noise_hugo(source):
	'''
	HUGO NOISE
	Transformation of picture into a noise. Works by shuffling pixels, i.e. by moving each pixel to another random position.
	Params:
		source	-- Image object to process
	'''
	
	source_pixelmap = source.load()
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	matrix = [[1 for j in range(source.size[1])] for i in range(source.size[0])]
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			a = -1
			b = -1
			while not ((a in range(result.size[0])) and (b in range(result.size[1])) and matrix[a][b] == 1):
				a = randrange(source.size[0])
				b = randrange(source.size[1])
			result_pixelmap[a,b] = source_pixelmap[x,y]
	return result

def t_filter_dante(source, threshold, background, foreground):
	'''
	DANTE FILTER
	Transformation of image into binary (double-color) graphics by color intensity.
	Params:
		source		-- Image object to process
		threshold	-- color intensity threshold for foreground color application (float, <0,1>)
		background	-- background color (pixels with color intensity <= threshold) (3D vector)
		foreground	-- foreground color (pixels with color intensity > threshold) (3D vector)
	'''
	
	# Aux. function for calculating absolute color intensity from its channels.
	def magnitude(color):
		return int(round(sqrt(sum([i**2 for i in color])/float(3 * 255**2))))

	source_pixelmap = source.load()
	result = Image.new(source.mode, source.size, background)
	result_pixelmap = result.load()
	
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			if magnitude(source_pixelmap[x,y]) > threshold:
				result_pixelmap[x,y] = foreground
	return result


def t_blend_color(source, blend_color, blend_level):
	'''
	BLEND WITH COLOR
	Overlays picture with set color.
	Params:
		source		-- Image object to process
		blend_color	-- overlay color (3D vector)
		blend_level	-- intensity of overlay (float <0,1>)
	'''
	
	# Aux. function to mix source pixel color with overlay color.
	def blend_pixel(source_pixel, blend_color, blend_level):
		return tuple([int(round(i)) for i in [(1 - blend_level)*source_pixel[i] + blend_level*blend_color[i] for i in range(3)]])
	
	source_pixelmap = source.load()
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			result_pixelmap[x,y] = blend_pixel(source_pixelmap[x,y], blend_color, blend_level)
	return result


def t_blend_image(source, blend_image, blend_level):
	'''
	BLEND WITH IMAGE
	Overlays picture with another picture. The overlay picture doesn't have to be the same size as source.
	Params:
		source		-- Image object to process
		blend_image	-- Image object to overlay source with
		blend_level	-- intensity of overlay (float <0,1>)
	'''
	
	# Aux. function to mix source pixel color with overlay color.
	def blend_pixel(source_pixel, blend_pixel, blend_level):
		return tuple([int(round(i)) for i in [(1 - blend_level)*source_pixel[i] + blend_level*blend_pixel[i] for i in range(3)]])
	
	source_pixelmap = source.load()
	blend_image_pixelmap = blend_image.load()
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			result_pixelmap[x,y] = blend_pixel(source_pixelmap[x,y], blend_image_pixelmap[x % blend_image.size[0], y % blend_image.size[1]], blend_level)
	return result


def t_blur(source, range_x, range_y):
	'''
	MATRIX BLUR
	Blurs each pixel by averaging it with surrounding pixels. It is possible to change different size of blur range in X and Y axis.
	Params:
		source	-- Image object to process
		range_x	-- range of blur on X axis
		range_y	-- range of blur on Y axis
	'''
	
	source_pixelmap = source.load()
	
	# Aux. function to average the colors of surrounding pixels.
	def average_pixel(source, coords, range_x, range_y):
		x, y = coords
		colors = list(source_pixelmap[x,y])
		pixel_count = 1
		for x1 in range(-1 * range_x, range_x + 1):
			for y1 in range(-1 * range_y, range_y + 1):
				try:
					current_color = source_pixelmap[x + x1, y + y1]
					for i in range(3):
						colors[i] += current_color[i]
					pixel_count += 1
				except IndexError:
					continue
		return tuple(map(lambda i: int(round(float(i)/pixel_count)), colors))
	
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			result_pixelmap[x,y] = average_pixel(source, (x,y), range_x, range_y)
	return result


##### EXPERIMENTAL
from collections import deque
def e_detect_objects(source, threshold):
	'''
	OBJECT DETECTION
	Tries to detect objects in an image based on their color difference using a graph traversal algorithm.
	Returns a new image in which detected objects are distinguished by color.
	Params:
		source		-- Image object to process
		threshold 	-- color difference that should be considered an edge between to objects
	'''
	
	# Aux. function calculating difference of two colors using a unit cube.
	def calculate_difference(color1, color2):
		return sqrt((color2[0] - color1[0])**2 + (color2[1] - color1[1])**2 + (color2[2] - color1[2])**2)/sqrt(3 * 255**2)

	def neighbors(pixel):
		result = list()
		for i in range(-1, 2):
			for j in range(-1, 2):
				if (i <> j) and (i in range(source.size[0])) and (j in range(source.size[1])):
					result.append((pixel[0] + i, pixel[1] + j))
		return result
	
	def average_color(pixelmap, pixels):
		return tuple([int(round(sum([pixelmap[p][i] for p in pixels])/float(len(pixels)))) for i in range(3)])
	
	source_pixelmap = source.load()
	queue = deque()
	queue.append((0,0))
	unprocessed = set()
	for x in range(source.size[0]):
		for y in range(source.size[1]):
			if (x,y) <> (0,0):
				unprocessed.add((x,y))
	objects = list()
	last_color = (-255,-255,-255)
	while len(queue) > 0:
		current = queue.popleft()
		if calculate_difference(last_color, source_pixelmap[current]) > threshold:
			objects.append(set())
		objects[-1].add(current)
		for neighbor in neighbors(current):
			if neighbor in unprocessed:
				unprocessed.remove(neighbor)
				if calculate_difference(source_pixelmap[current], source_pixelmap[neighbor]) <= threshold:
					queue.appendleft(neighbor)
				else:
					queue.append(neighbor)
		last_color = source_pixelmap[current]
	
	result = Image.new(source.mode, source.size, (0,0,0))
	result_pixelmap = result.load()
	
	for group in objects:
		color = average_color(source_pixelmap, group) 
		for pixel in group:
			result_pixelmap[pixel] = color
	return result

def e_solve_maze(source, space_color, wall_color, start_color, end_color, path_color):
	'''
	MAZE SOLVING
	Finds path in a maze defined by the source picture.
	Works by graph traversal: starts on a random 'walkable' pixel, finds the starting pixel, then finds the shortest path to the finish pixel (using BFS).
	Params:
		source		-- Image object to process
		space_color	-- color that the algorithm is allowed to route the path through
		wall_color	-- color of walls (pixels that the algorithm is not allowed to walk through)
		start_color	-- color of starting pixel
		end_color	-- color of finish pixel
		path_color	-- color with which to highlight the found path
	'''
	
	# Aux. function to find all the valid (unprocessed & walkable) neighbors of a pixel
	def neighbors(source, pixel):
		pixelmap = source.load()
		result = list()
		for x in range(-1, 2):
			for y in range(-1, 2):
				try:
					if (pixel[0] + x in xrange(source.size[0])) and (pixel[1] + y in xrange(source.size[1])) and (dist[pixel[0] + x][pixel[1] + y] == None) and (pixelmap[pixel[0] + x, pixel[1] + y] <> wall_color):
						result.append((pixel[0] + x, pixel[1] + y))
				except IndexError:
					pass
		return result
	
	source_pixelmap = source.load()
	start_pixel = (300, 300)
	if not (source_pixelmap[start_pixel] == start_color):
		while not (source_pixelmap[start_pixel] == space_color):
			start_pixel = tuple([randrange(0, source.size[i]) for i in range(2)])
	
	dist = [[None for y in range(source.size[1])] for x in range(source.size[0])]
	queue = deque()
	
	# search for the start pixel
	current = start_pixel
	queue.append(start_pixel)
	processed = set()
	dist[start_pixel[0]][start_pixel[1]] = 0
	while not (source_pixelmap[current] == start_color) and len(queue) > 0:
		for n in neighbors(source, current):
			queue.append(n)
			dist[n[0]][n[1]] = dist[current[0]][current[1]] + 1
		processed.add(current)
		while current in processed:
				current = queue.popleft()
	
	# we found the start
	start_pixel = current
	dist = [[None for y in range(source.size[1])] for x in range(source.size[0])]
	prev = dict()
	queue = deque([start_pixel])
	processed = set()
	dist[current[0]][current[1]] = 0
	while not (source_pixelmap[current] == end_color) and len(queue) > 0:
		for n in neighbors(source, current):
			queue.append(n)
			dist[n[0]][n[1]] = dist[current[0]][current[1]] + 1
			prev[n] = current
		processed.add(current)
		while current in processed:
			current = queue.popleft()
	
	end_pixel = current
	current = prev[end_pixel]
	# mark the found path
	while current <> start_pixel:
		source_pixelmap[current] = path_color
		current = prev[current]
	
	return source
