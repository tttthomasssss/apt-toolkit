#!/bin/bash

# Usage e.g. ./vectors.sh /home/thk22/_datasets/wikipedia/apt/wikipedia_clean_chunked_10000_lexicon_pos_2 /home/thk22/_datasets/wikipedia/vectors/wikipedia_lc_2_pos_vectors.tsv /home/thk22/code/apt-standalone_incl-norm.jar 2>&1 | tee /home/thk22/_logs/apt_pos_vectors_depth_2.log

# Lexicon Input
LEXICON_PATH=$1

# Vector Output
OUTPUT_PATH=$2

# Optionally allow supplying an alternative jar path
if [ -z "$3" ]; then
	JAR_PATH="/home/thk22/code/apt-standalone.jar"
else
	JAR_PATH="$3"
fi

echo "Using JAR at path=$JAR_PATH ..."

if [ -z "$4" ]; then
	echo "Creating Vectors from $LEXICON_PATH; Outputting everyting to $OUTPUT_PATH..."
	# Standard Count Vectors
	java -Xmx350g -jar $JAR_PATH vectors "$LEXICON_PATH" "$OUTPUT_PATH";
	echo "Finished creating Vectors! - Lexicon: $LEXICON_PATH, Vectors: $OUTPUT_PATH!"
else
	echo "Creating [COUNT-NORMALISED] Vectors from $LEXICON_PATH; Outputting everyting to $OUTPUT_PATH..."
	# Normalised Count Vectors
	java -Xmx350g -jar $JAR_PATH vectors "$LEXICON_PATH" "$OUTPUT_PATH" "-normalise";
	echo "Finished creating [COUNT-NORMALISED] Vectors! - Lexicon: $LEXICON_PATH, Vectors: $OUTPUT_PATH!"
fi
