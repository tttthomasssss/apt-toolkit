from argparse import ArgumentParser
import gzip
import logging
import os
import sys

from apt_toolkit.utils.base import path_utils

parser = ArgumentParser()
parser.add_argument('-a', '--action', type=str, required=True, help='action to be executed')
parser.add_argument('-ip', '--input-path', type=str, required=True, help='path to input conll files')
parser.add_argument('-op', '--output-path', type=str, required=True, help='output path')
parser.add_argument('-l', '--use-lemma', action='store_true', help='use lemma instead of original token')
parser.add_argument('-p', '--use-pos', action='store_true', help='use PoS Tag')


def convert_line(line, use_lemma=True, use_pos=False):
	parts = line.split('\t')
	if len(parts) == 6:
		[idx, word, lemma, pos, gov_idx, rel] = parts
		w = lemma if use_lemma else word
		if (use_pos):
			return '\t'.join([idx, w.lower() + '/' + pos, gov_idx, rel])
		else:
			return '\t'.join([idx, w.lower(), gov_idx, rel])
	elif len(parts) == 4:
		[idx, word, lemma, pos] = parts
		w = lemma if use_lemma else word
		if (use_pos):
			return '\t'.join([idx, w.lower() + '/' + pos])
		else:
			return '\t'.join([idx, w.lower()])
	elif len(parts) == 7:
		[idx, word, lemma, pos, ner, gov_idx, rel] = parts
		w = lemma if use_lemma else word
		if (use_pos):
			if (gov_idx == '_'): # FIXME: This check is not implemented for the other lengths
				return '\t'.join([idx, w.lower() + '/' + pos])
			else:
				return '\t'.join([idx, w.lower() + '/' + pos, gov_idx, rel])
		else:
			if (gov_idx == '_'):
				return '\t'.join([idx, w.lower()])
			else:
				return '\t'.join([idx, w.lower(), gov_idx, rel])
	else:
		logging.warning('[WARNING] - line of unhandled length: {}'.format(len(parts)))
		return line


def open_file(filename, mode='r'):
	if filename.endswith('.gz'):
		return gzip.open(filename, mode)
	else:
		return open(filename, mode)


def convert_file(input_filename, output_filename, use_lemma=True, use_pos=False):
	with open_file(input_filename) as incoming:
		with open_file(output_filename, 'w') as outgoing:
			sent = []
			for line in incoming:
				line = line.strip()
				if len(line) == 0:
					try:
						processed_sent = '\n'.join([convert_line(line, use_lemma, use_pos) for line in sent])
						outgoing.write(processed_sent + '\n\n')
					except Exception:
						logging.info('couldn\'t process sentence: ')
						logging.info('\n'.join(sent))
						logging.info('\n (ignoring)\n')
					sent = []
				else:
					sent.append(line)


def stanford_conll_conversion(input_path, output_path, use_lemma, use_pos):
	logging.info('input_directory={};\noutput_directory={};\nuse_lemma={};\nuse_pos={}'.format(input_path, output_path, use_lemma, use_pos))

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for f in os.listdir(input_path):
		logging.info('converting ' + f)
		convert_file(os.path.join(input_path, f), os.path.join(output_path, f), use_lemma=use_lemma, use_pos=use_pos)


if (__name__ == '__main__'):
	args = parser.parse_args()

	timestamped_foldername = path_utils.timestamped_foldername()
	log_path = os.path.join(path_utils.get_log_path(), timestamped_foldername)
	if (not os.path.exists(log_path)):
		os.makedirs(log_path)

	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='[%d/%m/%Y %H:%M:%S] - %p')
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.DEBUG)
	root_logger.addHandler(logging.StreamHandler(sys.stdout))
	file_handler = logging.FileHandler(os.path.join(log_path, 'conll_conversion_{}.log'.format(args.action)))
	root_logger.addHandler(file_handler)

	if (args.action == 'stanford_conll_conversion'):
		stanford_conll_conversion(input_path=args.input_path, output_path=args.output_path, use_lemma=args.use_lemma,
								  use_pos=args.use_pos)
	else:
		raise ValueError('Unknown Action: {}'.format(args.action))
