## Dependencies
	
The code relies on several 3rd party libraries:

* numpy
* scipy
* scikit-learn
* dill
* joblib
* nltk
* sqlitedict
	
In addition the following code (which has its own dependencies) is necessary for performing distributional inference:

* DiscoUtils: `https://github.com/tttthomasssss/DiscoUtils` or from the original repository: `https://github.com/mbatchkarov/DiscoUtils` (both should be fine, if in doubt the forked version by tttthomasssss should work)

----

## Installation

Apart from `DiscoUtils` which needs to be installed manually, all requirements as well as the codebase itself can be installed with:
	
	cd path/to/apt-toolkit
	pip install -e .

----

## Resources

Vectors from the EMNLP 2016 paper `Improving Sparse Word Representations with Distributional Inference for Semantic Composition` are in the subfolder `resources/vectors`.

The folder contains 5 files:

* `APT_ML2010_AN.dill.xz`: Composed AN vectors (using 1000 neighbours), keys are prefixed with `__CV_`.
* `APT_ML2010_NN.dill.xz`: Composed NN vectors (using 10 neighbours), keys are prefixed with `__CV_`
* `APT_ML2010_VO.dill.xz`: Composed VO vectors (using 50 neighbours), keys are prefixed with `__CV_`
* `APT_wikipedia_clean_shift-10.dill.xz`: Raw APT vectors, with an SPPMI shift of `log(10)`.
* `APT_wikipedia_clean_shift-40.dill.xz`: Raw APT vectors, with an SPPMI shift of `log(40)`.

All files have been compressed with [`xz`](http://imoverclocked.blogspot.co.at/2015/12/for-love-of-bits-stop-using-gzip.html). All files can be read with the supplied code by using `vector_utils.load_vector_cache()` (see below).

----

## Usage

#### Loading vectors:

	from apt_toolkit.utils import vector_utils
	
	vectors = vector_utils.load_vector_cache('path/to/vectors', filetype='dill') # Loads the higher-order dependency-typed vectors as a `dict` of `dicts`
	
#### Composing Vectors:
	
	from apt_toolkit.composition import mozart
	from apt_toolkit.utils import vector_utils

	# Load Vectors
	vectors = vector_utils.load_vector_cache('path/to/vectors', filetype='dill')
	
	noun_vector = vectors['quantity']
	adj_vector = vectors['large']
	
	# Suppose we want to compose the AN pair "large quantity" with a an "amod" relation between the adjective and the noun, we first need to offset the adjective by "amod" to align the feature spaces (doing so makes the adjective more look like a noun; the offset needs to be "amod" as the adjective has an inverse "amod" relation to its head noun)
	offset_vector = vector_utils.create_offset_vector(adj_vector, 'amod')
	
	# Now the two vectors can be composed
	composed_vector = mozart.intersect_apts(offset_vector, noun_vector)
	
#### Distributional Inference:
	
	from apt_toolkit.distributional_inference import distributional_inference
	from apt_toolkit.utils import vector_utils
	
	# Load Vectors
	vectors = vector_utils.load_vector_cache('path/to/vectors', filetype='dill')
	
	# Use the top 20 neighbours (using the `static top n` approach) to infer unobserved co-occurrence features for the noun "book"
	rich_book = distributional_inference.static_top_n(vectors=vectors, words=['book'], num_neighbours=20)