__author__ = 'thk22'
from datetime import datetime
import os
import pwd
import socket


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Unwrap, unwrap, unwrap...

# Extra handling for the disco machine
_DISCO_BASEPATHS = {
	'thk22': os.path.expanduser('~'),
	'juliewe': 'base/path',
	'rd263': 'base/path'
}

_DISCO_DATASET_PATHS = {
	'thk22': os.path.expanduser('~/_datasets'),
	'juliewe': 'dataset/path',
	'rd263': 'dataset/path'
}

_DISCO_OUT_PATHS = {
	'thk22': os.path.expanduser('~/_results'),
	'juliewe': 'results/path',
	'rd263': 'results/path'
}

_DISCO_LOG_PATHS = {
	'thk22': os.path.expanduser('~/_logs'),
	'juliewe': 'logs/path',
	'rd263': 'logs/path'
}

# Parent Folder of _everything_
_BASEPATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab'),
}

# Path to datasets
_DATASET_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/EpicDataShelf/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_datasets/'),
}

# Path to Results
_OUT_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab/_results'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_results'),
}

# Path to logs
_LOG_PATH = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab/_logs'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_logs'),
}

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


def get_base_path(fallback='/lustre/scratch/inf/thk22/code'):
	if (socket.gethostname() == 'disco'):
		return _DISCO_BASEPATHS[pwd.getpwuid(os.getuid()).pw_name]
	else:
		return _BASEPATH.get(socket.gethostname(), fallback) # Fallback on lustre path on cluster


def get_dataset_path(fallback='/lustre/scratch/inf/thk22/_datasets'):
	if (socket.gethostname() == 'disco'):
		return _DISCO_DATASET_PATHS[pwd.getpwuid(os.getuid()).pw_name]
	else:
		return _DATASET_PATH.get(socket.gethostname(), fallback) # Fallback on lustre path on cluster


def get_out_path(fallback='/lustre/scratch/inf/thk22/_results'):
	if (socket.gethostname() == 'disco'):
		return _DISCO_OUT_PATHS[pwd.getpwuid(os.getuid()).pw_name]
	else:
		return _OUT_PATH.get(socket.gethostname(), fallback) # Fallback on lustre path on cluster


def get_log_path(fallback='/lustre/scratch/inf/thk22/_logs'):
	if (socket.gethostname() == 'disco'):
		return _DISCO_LOG_PATHS[pwd.getpwuid(os.getuid()).pw_name]
	else:
		return _LOG_PATH.get(socket.gethostname(), fallback) # Fallback on lustre path on cluster


def get_external_base_path():
	return _EXTERNAL_BASE_PATH[socket.gethostname()] # fail loudly!!!


def get_external_dataset_path(fallback='/lustre/scratch/inf/thk22/_datasets/CACHE'):
	return _EXTERNAL_DATASET_PATH.get(socket.gethostname(), fallback)