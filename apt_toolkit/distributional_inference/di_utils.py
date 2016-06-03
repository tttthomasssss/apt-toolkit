__author__ = 'thk22'

def smooth_apt(smoothing_function, disco_vectors, apt, word, num_neighbours=100, **kwargs):

	pos = kwargs.pop('pos')
	if ('wordnet' in smoothing_function.__name__):
		words = [(word, pos)]
	else:
		words = [word]

	_, smoothed_apt = smoothing_function(disco_vectors, words, num_neighbours=num_neighbours, **kwargs)

	if (isinstance(smoothed_apt[word], dict)):
		return smoothed_apt[word] # Inverse transform already performed
	else:
		smoothed_cols = smoothed_apt[word].nonzero()[1]

		for col in smoothed_cols:
			feat = disco_vectors.columns[col]

			apt[feat] = smoothed_apt[word][0, col]

		return apt