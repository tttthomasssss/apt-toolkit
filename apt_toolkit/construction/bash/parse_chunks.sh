#!/bin/bash

# Usage e.g. ./parse_chunks.sh /home/thk22/_resources/stanford-corenlp-full-2015-04-20 /home/thk22/_datasets/wikipedia/apt/wikipedia_chunked_10000 /home/thk22/_datasets/wikipedia/apt/wikipedia_chunked_10000_parsed 2>&1 | tee ~/_logs/chunked_parse.log

CORENLP_PATH=$1 # Path to corenlp.sh
INPUT_PATH=$2 # Path to raw csv chunks
OUTPUT_PATH=$3 # Outfolder for parsed chunks

if [ -z $4 ]; then
	EXT="tsv"
else
	EXT=$4
fi

# Loop over chunks and parse them
find $INPUT_PATH -name "*.$EXT" | while read line; do
	INPUT_FILE=$(basename $line)
	echo "Parsing $INPUT_FILE..."
	$CORENLP_PATH/corenlp.sh -annotators tokenize,ssplit,pos,lemma,parse -file $INPUT_FILE -ParserAnnotator BasicDependenciesAnnotation -outputFormat conll -outputExtension .parsed -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory $OUTPUT_PATH
	echo "====================================================================="
done
