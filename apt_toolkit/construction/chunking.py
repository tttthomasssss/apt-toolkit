__author__ = 'thomas'
from argparse import ArgumentParser

import csv
import gzip
import logging
import os
import sys

from apt_toolkit.utils.base import path_utils

parser = ArgumentParser()
parser.add_argument('-a', '--action', type=str, required=True, help='action to be executed')
parser.add_argument('-i', '--input-file', type=str, required=True, help='input file to use')
parser.add_argument('-ip', '--input-path', type=str, required=True, help='path to input file')
parser.add_argument('-op', '--output-path', type=str, required=True, help='output path')
parser.add_argument('-s', '--chunk-size', type=int, default=10000, help='size of chunks')
parser.add_argument('-ot', '--output-file-template', type=str, default='chunk_{}.tsv', help='name template for the output file')


def split_tsv_file(input_path, input_file, output_path, output_file_template='chunk_{}', chunk_size=10000):
	csv.field_size_limit(sys.maxsize)

	curr_chunk = 0
	n_chunks = 1
	new_file = True

	with open(os.path.join(input_path, input_file), 'r') as csv_file:
		csv_reader = csv.reader(csv_file)

		for idx, full_line in enumerate(csv_reader):
			# Fucking first item is a fucking wikipedia id, which is fucking annoying
			line = list(map(lambda x: x.strip(), full_line[1:]))
			if (new_file):
				out_file = output_file_template.format(n_chunks)
				out_csv = open(os.path.join(output_path, out_file), 'w')

				csv_writer = csv.writer(out_csv)
				logging.info('Outputting to {}...'.format(out_file))

				new_file = False

			if (idx % 10000 == 0):
				logging.info('\tProcessed {} lines!'.format(idx))

			if (curr_chunk < chunk_size):
				csv_writer.writerow(line)
				curr_chunk += 1
			else:
				csv_writer.writerow(line)
				out_csv.close()
				n_chunks += 1
				curr_chunk = 0
				new_file = True

	logging.info('Finished!!')


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
	file_handler = logging.FileHandler(os.path.join(log_path, 'chunking_{}.log'.format(args.action)))
	root_logger.addHandler(file_handler)

	if (args.output_path is not None and not os.path.exists(args.output_path)):
		os.makedirs(args.output_path)

	if (args.action == 'split_tsv_vile'):
		split_tsv_file(input_path=args.input_path, input_file=args.input_file, output_path=args.output_path,
					   output_file_template=args.output_file_template, chunk_size=args.chunk_size)
	else:
		raise ValueError('Unknown Action: {}'.format(args.action))