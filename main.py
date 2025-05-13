import pandas as pd
import numpy as np
import torch
import nltk
import argparse
from pathlib import Path
from transformers import AutoTokenizer, BertModel
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import tensorflow_hub as hub
import logging
import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
tf.get_logger().setLevel('ERROR')
logging.getLogger('transformers').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_environment():
    """Initialize the environment and download required resources."""
    logging.info("Setting up environment...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set up NLTK data directory in the project folder
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Download NLTK data
    logging.info("Downloading NLTK resources...")
    try:
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {str(e)}")
        logging.error("Please run the following commands manually:")
        logging.error("python -m nltk.downloader stopwords")
        logging.error("python -m nltk.downloader punkt")
        sys.exit(1)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return device

# Define questions
QUESTIONS = {
    1: 'es adecuado que Laura le dedique paulatinamente más tiempo al trabajo y su desarrollo profesional que a la familia y a las otras dimensiones de su vida',
    2: 'respecto de los ingenieros que renuncian al proyecto debido al impacto generado, que decisión le parece más correcta'
}

# Constants
INPUT_FILE = 'data/sel_cons.csv'
PHASE = 1

class DataLoader:
    """Class to handle data loading and preprocessing."""
    
    def __init__(self, input_file, phase, question_number):
        self.input_file = input_file
        self.phase = phase
        self.question_number = question_number
        self.stop_words = set(stopwords.words('spanish'))
        
    def load_data(self):
        """Load and preprocess the dataset."""
        logging.info(f"Loading data from {self.input_file}...")
        
        # Check if file exists
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Load dataframe
        self.dataframe = pd.read_csv(self.input_file)
        initial_size = len(self.dataframe)
        logging.info(f"Loaded {initial_size} total responses")
        
        # Filter by phase and question
        self.dataframe = self.dataframe[
            (self.dataframe['phase'] == self.phase) & 
            (self.dataframe['df'] == self.question_number)
        ]
        filtered_size = len(self.dataframe)
        logging.info(f"Filtered to {filtered_size} responses (Phase {self.phase}, Question {self.question_number})")
        
        # Preprocess comments
        self._preprocess_comments()
        
        return self.dataframe.sample(50) 
    
    def _preprocess_comments(self):
        """Preprocess the comments in the dataframe."""
        logging.info("Preprocessing comments...")
        
        # Remove rows with empty comments
        self.dataframe.dropna(subset=['comment'], inplace=True)
        
        # Process comments
        self.dataframe['comment_processed'] = [
            x.replace('.', '').replace(',', '').lower() 
            for x in self.dataframe['comment']
        ]
        
        # Remove empty comments and duplicates
        self.dataframe = self.dataframe[
            self.dataframe['comment_processed'].str.len() > 0
        ].drop_duplicates()
        
        final_size = len(self.dataframe)
        logging.info(f"After preprocessing: {final_size} valid responses")
        
        # Log some statistics
        avg_length = self.dataframe['comment_processed'].str.len().mean()
        logging.info(f"Average comment length: {avg_length:.1f} characters")

class SemanticAnalyzer:
    def __init__(self, model_name, case_file, question_number, top_n=30):
        self.model_name = model_name
        self.case_file = case_file
        self.question_number = question_number
        self.top_n = top_n
        self.device = setup_environment()
        self.stop_words = set(stopwords.words('spanish'))
        
        logging.info(f"Initializing analyzer with model: {model_name}")
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        logging.info(f"Loading {self.model_name} model...")
        if self.model_name == 'beto':
            self.tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
            self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        elif self.model_name == 'use':
            self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        elif self.model_name == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        logging.info(f"{self.model_name} model loaded successfully")
            
    def _load_data(self):
        # Initialize data loader
        data_loader = DataLoader(INPUT_FILE, PHASE, self.question_number)
        self.dataframe = data_loader.load_data()
        
        # Load case text
        with open(self.case_file, 'r') as file:
            self.case_text = file.read()
        logging.info(f"Case text loaded from {self.case_file}")
            
        # Get question text
        self.question_text = QUESTIONS[self.question_number]
        logging.info(f"Selected question {self.question_number}: {self.question_text}")
        
    def _preprocess_text(self, text):
        return ' '.join([x.lower() for x in text.replace('.', '').replace(',', '').split() 
                        if x.lower() not in self.stop_words])
        
    def _get_embeddings(self, texts, is_query=False):
        logging.info(f"Generating embeddings using {self.model_name}...")
        if self.model_name == 'beto':
            return self._get_beto_embeddings(texts)
        elif self.model_name == 'use':
            return self._get_use_embeddings(texts)
        elif self.model_name == 'tfidf':
            return self._get_tfidf_embeddings(texts, is_query)
            
    def _get_beto_embeddings(self, texts):
        """Get BETO embeddings for texts."""
        if isinstance(texts, str):
            # Single text case (for query)
            tokens = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**tokens)
            return outputs['pooler_output'].numpy()
        else:
            # Multiple texts case (for comments)
            embeddings = []
            total = len(texts)
            for i, text in enumerate(texts, 1):
                if i % 10 == 0:  # Log more frequently since we're using a smaller sample
                    logging.info(f"Processing BETO embeddings: {i}/{total}")
                tokens = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**tokens)
                embeddings.append(outputs['pooler_output'].numpy())
            return embeddings
    
    def _get_use_embeddings(self, texts):
        """Get USE embeddings for texts."""
        if isinstance(texts, str):
            # Single text case (for query)
            return self.use_model([texts]).numpy()
        else:
            # Multiple texts case (for comments)
            return self.use_model(texts).numpy()
    
    def _get_tfidf_embeddings(self, texts, is_query=False):
        """Get TF-IDF embeddings for texts."""
        if not is_query:
            # For comments, fit and transform
            self.vectorizer = TfidfVectorizer()
            return self.vectorizer.fit_transform(texts)
        else:
            # For query, just transform using the fitted vectorizer
            return self.vectorizer.transform(texts)
    
    def analyze(self):
        # Load and preprocess data
        self._load_data()
        
        # Preprocess case and question
        logging.info("Preprocessing case and question text...")
        case_processed = self._preprocess_text(self.case_text)
        question_processed = self._preprocess_text(self.question_text)
        query_text = f"{question_processed} {case_processed}"
        
        # Get embeddings for comments and query
        comments = self.dataframe['comment_processed'].values
        if self.model_name == 'tfidf':
            # For TF-IDF, fit on comments first, then transform both
            comment_embeddings = self._get_embeddings(comments, is_query=False)
            query_embedding = self._get_embeddings([query_text], is_query=True)
        else:
            comment_embeddings = self._get_embeddings(comments)
            query_embedding = self._get_embeddings(query_text)
        
        # Calculate similarities
        logging.info("Calculating semantic similarities...")
        if self.model_name == 'tfidf':
            similarities = cosine_similarity(query_embedding, comment_embeddings).flatten()
        elif self.model_name == 'use':
            similarities = [cosine_similarity(query_embedding.reshape(1, -1), 
                                           comment_embedding.reshape(1, -1))[0][0] 
                          for comment_embedding in comment_embeddings]
        else:  # beto
            similarities = [cosine_similarity(query_embedding, comment_embedding)[0][0] 
                          for comment_embedding in comment_embeddings]
        
        # Convert similarities to numpy array
        similarities = np.array(similarities)
        
        # Get top N results
        top_indices = similarities.argsort()[-self.top_n:][::-1]
        top_comments = self.dataframe.iloc[top_indices].copy()
        
        # Add similarity scores to the results
        top_comments['similarity_score'] = similarities[top_indices]
        
        # Create output filename with model, question, and N
        output_filename = f"results/top{self.top_n}_{self.model_name}_q{self.question_number}.csv"
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        top_comments.to_csv(output_filename, index=False)
        
        # Log results summary
        logging.info(f"\nResults Summary:")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Question: {self.question_number}")
        logging.info(f"Top {self.top_n} responses saved to {output_filename}")
        logging.info(f"Similarity scores range: {min(similarities):.3f} to {max(similarities):.3f}")
        
        # Log sample of top results
        logging.info("\nTop 5 most similar responses:")
        for i, (_, row) in enumerate(top_comments.head().iterrows(), 1):
            logging.info(f"\n{i}. Score: {row['similarity_score']:.3f}")
            logging.info(f"Comment: {row['comment']}")
        
        return top_comments

def main():
    parser = argparse.ArgumentParser(description='Semantic Analysis of Educational Responses')
    parser.add_argument('-m', '--model', choices=['beto', 'use', 'tfidf'], required=True,
                      help='Model to use for analysis (beto, use, or tfidf)')
    parser.add_argument('-c', '--case', required=True,
                      help='Input file containing the case text')
    parser.add_argument('-q', '--question', type=int, choices=[1, 2], required=True,
                      help='Question number to analyze (1 or 2)')
    parser.add_argument('-n', '--topn', type=int, default=30,
                      help='Number of top responses to select (default: 30)')
    
    args = parser.parse_args()
    
    logging.info("Starting semantic analysis...")
    logging.info(f"Configuration: Model={args.model}, Question={args.question}, TopN={args.topn}")
    
    analyzer = SemanticAnalyzer(
        model_name=args.model,
        case_file=args.case,
        question_number=args.question,
        top_n=args.topn
    )
    
    results = analyzer.analyze()
    logging.info("Analysis completed successfully")

if __name__ == "__main__":
    main()


