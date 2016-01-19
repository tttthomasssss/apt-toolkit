__author__ = 'thk22'

from apt_toolkit.utils import vector_utils


def add_apts(vector_1, vector_2):
	composed_vector = {}
	for feat in (set(vector_1.keys()) | set(vector_2.keys())):
		composed_vector[feat] = vector_1.get(feat, 0.) + vector_2.get(feat, 0.)

	return composed_vector


def intersect_apts(vector_1, vector_2):
	composed_vector = {}
	for feat in (set(vector_1.keys()) & set(vector_2.keys())):
		composed_vector[feat] = vector_1[feat] + vector_2[feat]

	return composed_vector


def path_intersect_apts(vector_1, vector_2):
	composed_vector = {}
	path_set_v1 = set()
	path_set_v2 = set()

	for feat in set(vector_1.keys()):
		p, _ = vector_utils.split_path_from_word(feat)
		path_set_v1.add(p)
	for feat in set(vector_2.keys()):
		p, _ = vector_utils.split_path_from_word(feat)
		path_set_v2.add(p)
	path_set = path_set_v1 & path_set_v2

	for feat in (set(vector_1.keys()) | set(vector_2.keys())):
		p, _ = vector_utils.split_path_from_word(feat)
		if (p in path_set):
			composed_vector[feat] = vector_1.get(feat, 0.) + vector_2.get(feat, 0.)

	return composed_vector


def renormalise_vector(vector):
	total = sum(vector.values())

	for feat, val in vector.items():
		vector[feat] /= total

	return vector