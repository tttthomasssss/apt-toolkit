__author__ = 'thk22'
from datetime import datetime
import os
import socket


PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

_BASEPATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab'),
	'disco': os.path.expanduser('~')
}

_DATASET_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/EpicDataShelf/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_datasets/'),
	'disco': os.path.expanduser('~/_datasets')
}

_OUT_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab/_results'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_results'),
	'disco': os.path.expanduser('~/_results')
}

_LOG_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab/_logs'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_logs'),
	'disco': os.path.expanduser('~/_logs')
}

# So far, only disco supports external paths
_EXTERNAL_BASE_PATH = {
	'disco': '/mnt/external/thk22/'
}

_EXTERNAL_DATASET_PATH = {
	'disco': '/mnt/external/thk22/_datasets',
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/EpicDataShelf/tag-lab/_CACHE/'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_datasets/CACHE')
}


# incl timestamp: format='%d%m%Y_%H%M%S'
def timestamped_foldername(format='%d%m%Y'):
	return datetime.now().strftime(format)


def get_base_path():
	return _BASEPATH.get(socket.gethostname(), '/lustre/scratch/inf/thk22/code') # Fallback on lustre path on cluster


def get_dataset_path():
	return _DATASET_PATH.get(socket.gethostname(), '/lustre/scratch/inf/thk22/_datasets') # Fallback on lustre path on cluster


def get_out_path():
	return _OUT_PATH.get(socket.gethostname(), '/lustre/scratch/inf/thk22/_results') # Fallback on lustre path on cluster


def get_log_path():
	return _LOG_PATH.get(socket.gethostname(), '/lustre/scratch/inf/thk22/_logs') # Fallback on lustre path on cluster


def get_external_base_path():
	return _EXTERNAL_BASE_PATH[socket.gethostname()] # fail loudly!!!


def get_external_dataset_path():
	return _EXTERNAL_DATASET_PATH.get(socket.gethostname(), '/lustre/scratch/inf/thk22/_datasets/CACHE')