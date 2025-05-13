# Semantic Educational Analysis

This repository contains the implementation of a semantic analysis system for educational responses, as described in our paper: *"Low-Footprint NLP for Reducing Teacher’s Orchestration Load in Computer-Supported Case-Based Learning Environments"* submitted to the Journal of Universal Computer Science. The system uses three different models (BETO, Universal Sentence Encoder, and TF-IDF) to analyze and rank educational responses based on their semantic similarity to given questions and case studies.

## Features

- Support for three different semantic models:
  - BETO (Spanish BERT)
  - Universal Sentence Encoder
  - TF-IDF
- Command-line interface for easy usage
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

## Usage

The script can be run from the command line with the following arguments:

```bash
python main.py -m MODEL -c CASE_FILE -q QUESTION_NUMBER -o OUTPUT_FILE [-n TOP_N]
```

### Arguments

- `-m, --model`: Model to use for analysis (choices: 'beto', 'use', 'tfidf')
- `-c, --case`: Input file containing the case text
- `-q, --question`: Question number to analyze (1 or 2)
- `-n, --topn`: Number of top responses to select (default: 30)

### Available Questions

The system includes two predefined questions:

1. "¿Es adecuado que Laura le dedique paulatinamente más tiempo al trabajo y su desarrollo profesional que a la familia y a las otras dimensiones de su vida?"
2. "¿Respecto de los ingenieros que renuncian al proyecto debido al impacto generado, qué decisión le parece más correcta?"

### Example

```bash
python main.py -m tfidf -c data/case.txt -q 1 -n 30
```

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

### Output

The script generates a CSV file containing the top N responses ranked by semantic similarity to the question and case study.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{low_footprint_nlp,
  title={Low-Footprint NLP for Reducing Teacher’s Orchestration Load in Computer-Supported Case-Based Learning Environments},
  author={Claudio Alvarez,Andres Carvallo,Gustavo Zurita  Names},
  journal={Journal of Universal Computer Science},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
