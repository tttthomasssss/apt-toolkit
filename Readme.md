
#### Dependencies
	
_TBD_

#### Recommended Setup
	
_TBD_

#### Instructions for running for pre-processing a corpus, constructing an APT Lexicon and creating vectors

_These instructions pre-supposes that the base file is in `tsv` format and contains 1 article/book/document per line._
_Further note, the pipeline currently is only tested with the `Stanford Parser`._


* **Splitting the full file into chunks**

	see `construction/chunking.py`
	
	_Usage as script:_
	
			python -m construction.chunking --action split_tsv_file --input-path path/to/file --input-file the_file.tsv --output-path path/to/output --output-file-template the_file_chunk_{}.tsv --chunk-size 1000
	
	_Usage from code:_
		
		from apt_toolkit.construction import chunking
		chunking.split_tsv_file(input_path='path/to/file', input_file='the_file.tsv', output_path='path_to_output',
								output_file_template='the_file_chunk_{}.tsv', chunk_size=1000)

* **Parsing the chunks**

	see `construction/bash/parse_chunks.sh`
	
	_Usage:_
		
		./parse_chunks.sh /path/to/stanford-corenlp path/to/chunks path/to/output
		
* **Converting the output `conll` files to APT compatible `conll` files**
	
	see `construction/bash/convert_stanford_conll_to_apt_conll.sh`
	
	_Comments:_
	
	This script offers a little more (in)convenience in that it offers whether or not to use PoS Tags and whether or not to use the lemma. Basic usage outputs 4 different flavours of the input `conll` file:
	
	1. Lemma & PoS Tag
	2. Lemma & no PoS Tag
	3. original Token & PoS Tag
	4. original Token & no PoS Tag
	
	By passing `start` and `stop` options (see `Advanced Usage` below), it can be controlled which and how many of the above options should be executed. Note that it currently is only possible to execute ranges (e.g. 1-3; 2-4, ...) and not individual options (e.g. _only_ 1 and 3).
	
	_Usage:_
	
		./convert_stanford_conll_to_apt_conll.sh path/to/chunked_and_parsed path/to/output
		
	_Advanced Usage (e.g. only execut options 2 (Lemma & no PoS Tag) to 3 (original Token & PoS Tag)):_
	
		./convert_stanford_conll_to_apt_conll.sh path/to/chunked_and_parsed path/to/output 2 3

* **Construct the APT Lexicon**
	
	see `construction/bash/chunked_construct.sh`
	
	_Usage:_ 
		
		./chunked_construct.sh path/to/apt_compatible_conll_files path/to/output 2 parsed
		
* **Create Vectors from Lexicon**
	
	see `vectorisation/bash/vectors.sh`
	
	_Comments:_
	
	Pass `-normalise` as the last argument if you want count-normalised vectors (e.g.: `./vectors.sh path/to/apt_lexicon path/to/output_vectors.tsv path/to/apt_jarfile -normalise`)
	
	_Usage:_
	
		./vectors.sh path/to/apt_lexicon path/to/output_vectors.tsv path/to/apt_jarfile

#### Resources
	
_TBD_