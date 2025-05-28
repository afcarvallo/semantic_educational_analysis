# Semantic Educational Analysis

This repository contains the implementation of a semantic analysis system for educational responses, as described in our paper: *"Low-Footprint NLP for Reducing Teacher's Orchestration Load in Computer-Supported Case-Based Learning Environments"* submitted to the Journal of Universal Computer Science. The system uses three different models (BETO, Universal Sentence Encoder, and TF-IDF) to analyze and rank educational responses based on their semantic similarity to given questions and case studies.

## Features

- Support for three different semantic models:
  - BETO (Spanish BERT)
  - Universal Sentence Encoder
  - TF-IDF
- Ensemble analysis combining all three models with customizable weights
- Configurable number of top responses
- Support for Spanish language processing
- Automatic stopwords removal and text preprocessing
- Predefined questions for analysis
- Fixed input dataset (Phase 1 responses)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/semantic_educational_analysis.git
cd semantic_educational_analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Hugging Face Authentication

To use the BETO model, you need to authenticate with Hugging Face. Follow these steps:

1. Create a Hugging Face account at https://huggingface.co/join if you don't have one already.

2. Get your access token:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name and select "read" role
   - Copy the generated token

3. Login using the Hugging Face CLI:
```bash
huggingface-cli login
```
When prompted, paste your access token.

Alternatively, you can set the token as an environment variable:
```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

## Usage

The script can be run from the command line with the following arguments:

```bash
python main.py -m MODEL -c CASE_FILE -q QUESTION_NUMBER [-n TOP_N] [--beto-weight WEIGHT] [--use-weight WEIGHT] [--tfidf-weight WEIGHT]
```

### Arguments

- `-m, --model`: Model to use for analysis (choices: 'beto', 'use', 'tfidf', 'ensemble')
- `-c, --case`: Input file containing the case text
- `-q, --question`: Question number to analyze (1 or 2)
- `-n, --topn`: Number of top responses to select (default: 30)
- `--beto-weight`: Weight for BETO model in ensemble (default: 0.333333)
- `--use-weight`: Weight for USE model in ensemble (default: 0.333333)
- `--tfidf-weight`: Weight for TF-IDF model in ensemble (default: 0.333333)

### Available Questions

The system includes two predefined questions:

1. "¿Es adecuado que Laura le dedique paulatinamente más tiempo al trabajo y su desarrollo profesional que a la familia y a las otras dimensiones de su vida?"
2. "¿Respecto de los ingenieros que renuncian al proyecto debido al impacto generado, qué decisión le parece más correcta?"

### Examples

1. Using a single model:
```bash
python main.py -m tfidf -c data/case.txt -q 1 -n 30
```

2. Using ensemble with default weights (equal weights for all models):
```bash
python main.py -m ensemble -c data/case.txt -q 1 -n 30
```

3. Using ensemble with custom weights:
```bash
python main.py -m ensemble -c data/case.txt -q 1 -n 30 --beto-weight 0.2 --use-weight 0.5 --tfidf-weight 0.3
```

### Output

The script generates a CSV file containing the top N responses ranked by semantic similarity to the question and case study. When using ensemble mode, the output includes:

- Individual scores from each model (`beto_score`, `use_score`, `tfidf_score`)
- Combined ensemble score (`ensemble_score`)
- All scores are normalized and weighted according to the specified weights

### Input File Formats

1. Case File (`case.txt`):
   - Plain text file containing the case study
   
2. Responses Dataset:
   - Located at `data/sel_cons.csv`
   - Contains responses 
   - Required columns:
     - `comment`: The response text
     - `phase`: Phase number (only Phase 1 is used)
     - `df`: Question number (1 or 2) 


### Performance Analysis with Valgrind

You can use Valgrind to analyze memory usage and performance of the different models. Here's how to use it:

1. Install Valgrind:
```bash
# On Ubuntu/Debian
sudo apt-get install valgrind

# On macOS (using Homebrew)
brew install valgrind

# On CentOS/RHEL
sudo yum install valgrind
```

2. Run memory analysis:
```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt python main.py -m MODEL -c data/case.txt -q 1
```

3. Run performance analysis:
```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out python main.py -m MODEL -c data/case.txt -q 1
```

4. Analyze the results:
```bash
# For memory analysis
cat valgrind-out.txt

# For performance analysis (requires KCachegrind)
kcachegrind callgrind.out
```

Example output interpretation:
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 0 bytes in 0 blocks
==12345==   total heap usage: 1,234 allocs, 1,234 frees, 123,456 bytes allocated
```

Performance metrics to look for:
- Memory leaks (if any)
- Heap usage
- CPU time per model
- Number of allocations/deallocations
- Cache misses

Note: When running Valgrind with Python, you might need to use the `--suppressions` flag to suppress Python-specific warnings. You can generate a suppression file using:
```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --gen-suppressions=all python main.py -m MODEL -c data/case.txt -q 1 > python.supp
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{low_footprint_nlp,
  title={Low-Footprint NLP for Reducing Teacher's Orchestration Load in Computer-Supported Case-Based Learning Environments},
  author={Claudio Alvarez,Andres Carvallo,Gustavo Zurita},
  journal={Journal of Universal Computer Science},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
