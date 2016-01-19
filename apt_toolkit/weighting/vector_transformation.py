__author__ = 'thk22'
import argparse
import collections
import csv
import dill
import logging
import math
import os
import sys

from apt_toolkit.utils import vector_utils
from apt_toolkit.utils.base import path_utils

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--action', type=str, required=True, help='action to be executed')
parser.add_argument('-i', '--input-file', type=str, required=True, help='input file')
parser.add_argument('-p', '--input-path', type=str, default=os.path.join(path_utils.get_external_dataset_path(), 'wikipedia', 'vectors'), help='default path to input file')
parser.add_argument('-o', '--output-path', type=str, default=os.path.join(path_utils.get_external_dataset_path(), 'wikipedia', 'transformed_vectors'), help='output path')
parser.add_argument('-c', '--cache-intermediary-results', action='store_true', help='cache intermediary results')
parser.add_argument('-cf', '--count-cache-file-name', type=str, default='wikipedia', help='name of count cache file')
parser.add_argument('-cp', '--count-cache-base-path', type=str, default=os.path.join(path_utils.get_external_dataset_path(), 'wikipedia', 'vectors'), help='base path of count cache')
parser.add_argument('-e', '--experiment-id', type=int, default=-1, help='experiment id to run')
parser.add_argument('-lf', '--logging-frequency', type=int, default=1000, help='logging frequency inside the ppmi loop')
parser.add_argument('-xcv', '--exclude-composed-vectors', action='store_true', default=True, help='exclude composed vectors from contributing to the ppmi counts')
parser.add_argument('-cvp', '--composed-vector-prefix', type=str, default='__CV_', help='key prefix of composed vectors')
parser.add_argument('-f', '--force-rewrite', action='store_true', help='force to rewrite the vectors if they already exist.')


def _pmi_standard(w_T_w_prime, w_T_star, star_T_w, star_T_star, w_star_star, star_star_w_prime, w_star_w_prime, sum_T,
				  sum_w, sum_w_prime, cds, mathlog):
	return (mathlog(w_T_w_prime) - mathlog(w_T_star)) - (cds * (mathlog(star_T_w) - mathlog(star_T_star)))


def _pmi_interaction_information(w_T_w_prime, w_T_star, star_T_w, star_T_star, w_star_star, star_star_w_prime, w_star_w_prime, sum_T,
				  sum_w, sum_w_prime, cds, mathlog):
	weight = (w_T_w_prime / star_T_w) * (star_T_w / star_T_star) * (star_T_star / sum_T) * cds

	pmi = (mathlog(w_T_w_prime) - (cds * mathlog(star_T_w))) - (mathlog(w_T_star) - (cds * mathlog(star_T_star)))

	return weight * pmi


def _pmi_specific_interaction_information(w_T_w_prime, w_T_star, star_T_w, star_T_star, w_star_star, star_star_w_prime, w_star_w_prime, sum_T,
				  sum_w, sum_w_prime, cds, mathlog):
	return (cds * (mathlog(w_star_w_prime) - mathlog(star_star_w_prime)) + (mathlog(w_T_star) - mathlog(star_T_star))) - \
		   (((mathlog(w_star_star) - mathlog(sum_w)) + (mathlog(w_T_w_prime) - (cds * mathlog(star_T_w)))))


def _pmi_total_correlation(w_T_w_prime, w_T_star, star_T_w, star_T_star, w_star_star, star_star_w_prime, w_star_w_prime, sum_T,
				  sum_w, sum_w_prime, cds, mathlog):
	return ((mathlog(w_T_w_prime) - (cds * mathlog(star_T_w))) + (cds * (mathlog(star_T_w) - mathlog(star_T_star)))) - \
		   ((mathlog(w_star_star) - mathlog(sum_w)) + (cds * (mathlog(star_star_w_prime) - mathlog(sum_w_prime))))


def count_vectors(vector_in_file, out_base_path, cache_file_name='wikipedia', cache=False,
				  exclude_composed_vectors=True, composed_vector_prefix='__CV_'):
	logging.info('Starting counting...')
	universal_deps = open(os.path.join(path_utils.get_dataset_path(), 'stanford_universal_deps.csv'), 'r').read().strip().split(',')

	norm = '_norm' if 'norm' in vector_in_file else ''

	logging.info('Loading cache from file={}...'.format(vector_in_file))
	vectors = vector_utils.load_vector_cache(vector_in_file)
	logging.info('Successfully loaded cache!')

	dep_paths = collections.defaultdict(int)
	w_star_star = collections.defaultdict(int)
	star_star_w_prime = collections.defaultdict(int)
	word_path_event = collections.defaultdict(lambda: collections.defaultdict(int))
	path_word_prime_event = collections.defaultdict(lambda: collections.defaultdict(int))
	w_star_w_prime = collections.defaultdict(lambda: collections.defaultdict(int))
	word_path_word_prime_event = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))

	for idx, (entry, feat_dict) in enumerate(vectors.items(), 1):
		if (idx % 1000 == 0): logging.info('\tProcessing line {}...'.format(idx))
		if (not (exclude_composed_vectors and entry.startswith(composed_vector_prefix))):
			for event, freq in feat_dict.items():
				path, word_cooccurrence = vector_utils._split_path_from_word(event, universal_deps)

				# #(<*, T, *>)
				dep_paths[path] += 1

				# #(<w, *, *>)
				w_star_star[entry] += 1

				# #(<*, *, w'>)
				star_star_w_prime[word_cooccurrence] += 1

				# #(<w, *, w'>)
				w_star_w_prime[entry][word_cooccurrence] += 1

				# #(<w, T, *>)
				word_path_event[entry][path] += 1

				# #(<*, T, w'>)
				path_word_prime_event[path][word_cooccurrence] += 1

				# #(<w, T, w'>)
				word_path_word_prime_event[entry][path][word_cooccurrence] = freq
		else:
			logging.info('Excluding composed vector: {}'.format(entry))

	# Collect sums
	sums = {
		'sum_T': sum(dep_paths.values()),
		'sum_w': sum(w_star_star.values()),
		'sum_w_prime': sum(star_star_w_prime.values())
	}
	logging.info('Finished counting!')
	# Cache all count files (you never know when we might need them again...)
	if (cache):
		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_star_T_star.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(dep_paths, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_w_T_star.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(word_path_event, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_star_T_w_prime.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(path_word_prime_event, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_w_T_w_prime.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(word_path_word_prime_event, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_w_star_star.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(w_star_star, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_star_star_w_prime.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(star_star_w_prime, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_w_star_w_prime.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(w_star_w_prime, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

		out_path = os.path.join(out_base_path, ''.join([cache_file_name, norm, '_vectors_sums.dill']))
		logging.info('Caching count file: {}'.format(out_path))
		dill.dump(sums, open(out_path, 'wb'))
		logging.info('Cached file: {}!'.format(out_path))

	return (dep_paths, word_path_event, path_word_prime_event, word_path_word_prime_event, w_star_star,
			star_star_w_prime, w_star_w_prime, sums)


def ppmi_vectors(vector_in_file, out_base_path, cds=1., k=1., cache=False, count_fn=count_vectors,
				 count_cache_file_name='wikipedia', count_cache_base_path=os.path.join(path_utils.get_external_dataset_path(), 'wikipedia', 'vectors'),
				 logging_freq=1000, pmi_mode='standard', exclude_composed_vectors=True, composed_vector_prefix='__CV_', force_rewrite=False):

	# Path wanking
	norm = '_norm' if 'norm' in vector_in_file else ''

	pmi_suffix = '_sppmi' if k > 1. else '_pmi'
	if (cds > 1. or cds < 1.):
		pmi_suffix = '{}_cds-{}'.format(pmi_suffix, cds)

	pmi_mode_suffix = '_{}'.format(pmi_mode)

	if (vector_in_file.endswith('.joblib')):
		parts = vector_in_file.split('.joblib')
		splitter = ''
	elif ('_vectors' in vector_in_file):
		parts = vector_in_file.split('_vectors')
		splitter = '_vectors'

	logging.info('Split input filename into parts: {}; splitter={}...'.format(parts, splitter))

	if ('/' in parts[0]):
		vector_out_file = ''.join([os.path.split(parts[0])[1], norm, pmi_mode_suffix, pmi_suffix, splitter, parts[1]]) + '.dill'
	else:
		vector_out_file = ''.join([parts[0], norm, pmi_mode_suffix, pmi_suffix, splitter, parts[1]]) + '.dill'

	logging.info('vector_in_file={};\nvector_out_file={}\nvector_out_path={}\ncount_cache_base_path={}'.format(vector_in_file, vector_out_file, out_base_path, count_cache_base_path))

	# Check if the files already exist, if so ignore or rewrite them
	if (force_rewrite or not os.path.exists(os.path.join(out_base_path, vector_out_file))):

		# PMI mode function
		logging.info('using pmi_mode={}...'.format(pmi_mode))
		if (pmi_mode == 'interaction_information'): # Van De Cruys (2011); eqn. 5
			pmi_mode_fn = _pmi_interaction_information
		elif (pmi_mode == 'specific_interaction_information'): # Van De Cruys (2011); eqn. 8
			pmi_mode_fn = _pmi_specific_interaction_information
		elif (pmi_mode == 'total_correlation'): # Van De Cruys (2011); eqn 11
			pmi_mode_fn = _pmi_total_correlation
		else:
			pmi_mode_fn = _pmi_standard

		# Check if cache exists
		logging.info('Checking if cache exists at path={}...'.format(count_cache_base_path))
		star_T_star_path 		= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_star_T_star.dill']))
		w_T_star_path 			= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_w_T_star.dill']))
		star_T_w_prime_path 	= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_star_T_w_prime.dill']))
		w_T_w_prime_path 		= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_w_T_w_prime.dill']))
		w_star_star_path		= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_w_star_star.dill']))
		star_star_w_prime_path	= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_star_star_w_prime.dill']))
		w_star_w_prime_path		= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_w_star_w_prime.dill']))
		sums_path				= os.path.join(count_cache_base_path, ''.join([count_cache_file_name, norm, '_vectors_sums.dill']))

		if (not os.path.exists(star_T_star_path) 		and
			not os.path.exists(w_T_star_path)			and
			not os.path.exists(star_T_w_prime_path)		and
			not os.path.exists(w_T_w_prime_path)		and
			not os.path.exists(w_star_star_path)		and
			not os.path.exists(star_star_w_prime_path)	and
			not os.path.exists(w_star_w_prime_path)		and
			not os.path.exists(sums_path)):

			logging.info('No cache files found, starting counting...')
			(dep_paths, word_path_event, path_word_prime_event, word_path_word_prime_event,
			 p_w, p_w_prime, p_word_event, sums_dict) = count_fn(vector_in_file,
																			  out_base_path=count_cache_base_path,
																			cache=cache,
																			cache_file_name=count_cache_file_name)
			logging.info('Finished counting and caching!')
		else:
			logging.info('Found cached files, loading cache...')
			logging.info('\tLoading {}...'.format(star_T_star_path))
			dep_paths = dill.load(open(star_T_star_path, 'rb'))
			logging.info('\tLoading {}...'.format(w_T_star_path))
			word_path_event = dill.load(open(w_T_star_path, 'rb'))
			logging.info('\tLoading {}...'.format(star_T_w_prime_path))
			path_word_prime_event = dill.load(open(star_T_w_prime_path, 'rb'))
			logging.info('\tLoading {}...'.format(w_T_w_prime_path))
			word_path_word_prime_event = dill.load(open(w_T_w_prime_path, 'rb'))
			logging.info('Cache was loaded successfully!')
			logging.info('\tLoading {}...'.format(w_star_star_path))
			p_w = dill.load(open(w_star_star_path, 'rb'))
			logging.info('Cache was loaded successfully!')
			logging.info('\tLoading {}...'.format(star_star_w_prime_path))
			p_w_prime = dill.load(open(star_star_w_prime_path, 'rb'))
			logging.info('Cache was loaded successfully!')
			logging.info('\tLoading {}...'.format(w_star_w_prime_path))
			p_word_event = dill.load(open(w_star_w_prime_path, 'rb'))
			logging.info('Cache was loaded successfully!')
			logging.info('\tLoading {}...'.format(sums_path))
			sums_dict = dill.load(open(sums_path, 'rb'))
			logging.info('Cache was loaded successfully!')

		# Performance!!!
		mathlog = math.log

		# perform ppmi
		transformed_vectors = collections.defaultdict(lambda: collections.defaultdict(float))
		sum_T = sums_dict['sum_T']
		sum_w = sums_dict['sum_w']
		sum_w_prime = sums_dict['sum_w_prime']
		logging.info('Starting ppmi-ing...')
		for idx, w in enumerate(word_path_word_prime_event.keys(), 1):
			if (idx % logging_freq == 0): logging.info('PPMI-ing line {}'.format(idx))
			for jdx, T in enumerate(word_path_word_prime_event[w].keys(), 1):
				if (jdx % logging_freq == 0): logging.info('\tPPMI-ing path {}'.format(jdx))
				for kdx, w_prime in enumerate(word_path_word_prime_event[w][T].keys(), 1):
					if (kdx % logging_freq == 0): logging.info('\t\tPPMI-ing event {}'.format(kdx))
					w_T_w_prime = word_path_word_prime_event[w][T][w_prime]
					star_T_star = dep_paths[T]

					w_T_star = word_path_event[w][T]
					star_T_w = path_word_prime_event[T][w_prime]

					w_star_star = p_w[w]
					star_star_w_prime = p_w_prime[w_prime]
					w_star_w_prime = p_word_event[w][w_prime]

					# %timeit in ipython gave the result that doing a single log with divisions/multiplications is a lot
					# faster than multiple logs with addition/subtraction (although the second one might be numerically stabler)
					# pmi = np.log((star_T_star * w_T_w_prime) / (w_T_star * star_T_w))
					# pmi = (np.log(star_T_star) + np.log(w_T_w_prime)) - (np.log(w_T_star) + np.log(star_T_w))
					#ppmi = max((np.log(star_T_star) + np.log(w_T_w_prime)) - (np.log(w_T_star) + np.log(star_T_w)), 0)
					#
					# Use `math.log()` when dealing with scalars (much, much faster); also using an if instead of `max` is faster
					#ppmi = max((mathlog(star_T_star) + mathlog(w_T_w_prime)) - (mathlog(w_T_star) + mathlog(star_T_w)), 0)
					# PMI
					#logging.info('w_t_w_prime={}; w_T_star={}; star_T_w={}; star_T_star={}; cds={}'.format(w_T_w_prime, w_T_star, star_T_w, star_T_star, cds))
					#pmi = (mathlog(w_T_w_prime) - mathlog(w_T_star)) - (cds * (mathlog(star_T_w) - mathlog(star_T_star)))
					pmi = pmi_mode_fn(w_T_w_prime, w_T_star, star_T_w, star_T_star, w_star_star, star_star_w_prime,
									  w_star_w_prime, sum_T, sum_w, sum_w_prime, cds, mathlog)

					# Shift
					pmi -= mathlog(k)

					# 0 threshold, aka (S)PPMI
					ppmi = pmi if pmi >= 0 else 0

					event = ':'.join([T, w_prime])
					transformed_vectors[w][event] = ppmi

		logging.info('Starting vector dump to path={}...'.format(os.path.join(out_base_path, vector_out_file)))
		dill.dump(transformed_vectors, open(os.path.join(out_base_path, vector_out_file), 'wb'))
		logging.info('Successfully dumped weight transformed vectors!')
	else:
		logging.info('File {} at path {} already exists!'.format(vector_out_file, out_base_path))


if (__name__ == '__main__'):
	args = parser.parse_args()

	timestamped_foldername = path_utils.timestamped_foldername()
	log_path = os.path.join(path_utils.get_log_path(), timestamped_foldername)
	if (not os.path.exists(log_path)):
		os.makedirs(log_path)

	log_formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='[%d/%m/%Y %H:%M:%S] - %p')
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.DEBUG)

	file_handler = logging.FileHandler(os.path.join(log_path, 'vector_transformation_exp_id-{}.log'.format(args.experiment_id)))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(log_formatter)
	root_logger.addHandler(console_handler)

	vector_in_file = os.path.join(args.input_path, args.input_file)

	# Collect experiments
	experiments = []
	with open(os.path.join(PROJECT_PATH, 'resources', 'vector_transformation', 'transformation_setup.csv'), 'r') as csv_file:
		csv_reader = csv.reader(csv_file)
		# Collect Experiments
		for exp in csv_reader:
			experiments.append(exp)

	# Do stuff
	if (args.action == 'cache'):
		if (not os.path.exists(args.output_path)):
			os.makedirs(args.output_path)
		logging.info('Starting only counting and caching...')
		count_vectors(vector_in_file, out_base_path=args.count_cache_base_path, cache_file_name=args.count_cache_file_name,
					  cache=True)
		logging.info('Finished counting and caching!')
	elif (args.action == 'pmi'):
		logging.info('Starting pmi transformation...')
		exp = experiments[args.experiment_id-1]

		# Context Type subpath
		ctx_sub_path = os.path.split(args.input_path)[1]

		# Check path exists
		out_path = os.path.join(args.output_path, ctx_sub_path)
		if (not os.path.exists(out_path)):
			os.makedirs(out_path)

		logging.info('Experiment config: cds={}, k={}, out_path={}'.format(float(exp[0]), float(exp[1]), out_path))
		ppmi_vectors(vector_in_file, out_base_path=out_path, cds=float(exp[0]), k=float(exp[1]),
					 cache=args.cache_intermediary_results, count_cache_file_name=args.count_cache_file_name,
					 count_cache_base_path=args.count_cache_base_path, logging_freq=args.logging_frequency,
					 pmi_mode=exp[2], exclude_composed_vectors=args.exclude_composed_vectors, composed_vector_prefix=args.composed_vector_prefix,
					 force_rewrite=args.force_rewrite)
		logging.info('Finished pmi transformation!')
	elif (args.action == 'pmi_batch'):
		logging.info('Starting pmi batch transformation...')
		for exp in experiments:

			# Context Type subpath
			ctx_sub_path = os.path.split(args.input_path)[1]

			# Check path exists
			out_path = os.path.join(args.output_path, ctx_sub_path)
			if (not os.path.exists(out_path)):
				os.makedirs(out_path)

			logging.info('Experiment config: cds={}, k={}, out_path={}'.format(float(exp[0]), float(exp[1]), out_path))

			ppmi_vectors(vector_in_file, out_base_path=out_path, cds=float(exp[0]), k=float(exp[1]),
						 cache=args.cache_intermediary_results, count_cache_file_name=args.count_cache_file_name,
						 count_cache_base_path=args.count_cache_base_path, logging_freq=args.logging_frequency,
						 pmi_mode=exp[2], exclude_composed_vectors=args.exclude_composed_vectors,
						 composed_vector_prefix=args.composed_vector_prefix, force_rewrite=args.force_rewrite)

			logging.info('Finished transformation!')
		logging.info('Finished pmi batch transformation!')