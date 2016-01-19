#!/bin/bash

# Usage e.g.  ./chunked_construct.sh /home/thk22/_datasets/wikipedia/apt_converted_lemma-True_pos-True /home/thk22/_datasets/wikipedia/apt/wikipedia_clean_chunked_10000_lexicon_pos_2 2 parsed 2>&1 | tee /home/thk22/_logs/apt_construct_clean_pos_depth_2.log

INPUT_PATH=$1 # Folder with APT compatible conll parses
OUTPUT_PATH=$2 # Lexicon Path
DEPTH=$3 # Dependency Order

if [ -z $4 ]; then # File extension to look for
	EXT="parsed"
else
	EXT=$4
fi

if [ -z $5 ]; then # Path to APT jarfile
	JAR_PATH="/home/thk22/code/apt-standalone.jar"
else
	JAR_PATH=$6
fi

find $INPUT_PATH -name "*.$EXT" | while read line; do
	INPUT_FILE=$(basename $line)
	echo  "Processing $INPUT_FILE..."
	java -Xmx350g -jar $JAR_PATH construct $OUTPUT_PATH -depth $DEPTH "$INPUT_PATH/$INPUT_FILE";
	echo "Finished processing $INPUT_FILE!"
done
