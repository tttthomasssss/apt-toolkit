#!/bin/bash

# Usage e.g. ./convert_stanford_conll_to_apt_conll.sh /home/thk22/_datasets/wikipedia/apt/wikipedia_clean_chunked_10000_parsed/ /home/thk22/_datasets/wikipedia

USE_LEMMA[1]="-l"
USE_LEMMA[2]="-l"
USE_LEMMA[3]=""
USE_LEMMA[4]=""

USE_POS[1]="-p"
USE_POS[2]=""
USE_POS[3]="-p"
USE_POS[4]=""

INPUT_DIRECTORY=$1 # Path to stanford-parsed conll files
OUTPUT_BASEPATH=$2 # Output path of APT compatible conll files

if [ -z "$3" ]; then
	START=1
else
	START=$3
fi

if [ -z "$4" ]; then
	STOP=4
else
	STOP=$4
fi

for i in `seq $START 1 $STOP`;
do
	echo "Converting with options: use_lemma=${USE_LEMMA[$i]}; use_pos=${USE_POS[$i]}..."
	OUTPUT_DIRECTORY="$OUTPUT_BASEPATH/apt_converted_lemma-${USE_LEMMA[$i]}_pos-${USE_POS[$i]}"
	if [ ! -d "$OUTPUT_DIRECTORY" ]; then
		mkdir -p "$OUTPUT_DIRECTORY"
	fi
	python -m apt_toolkit.construction.conll_conversion --action stanford_conll_conversion --input-path $INPUT_DIRECTORY --output-path $OUTPUT_DIRECTORY ${USE_LEMMA[$i]} ${USE_POS[$i]}
	echo "Finished!"
	echo "===================================================================================================="
done