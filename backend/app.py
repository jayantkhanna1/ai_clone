from datetime import datetime
import logging
import pickle
import docx
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
import re
import string
from flask_cors import cross_origin
from collections import Counter, defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://jayantkhanna1.github.io", "http://localhost:3000", "*", "https://*.jayantkhanna.in"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})


# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_PATH = 'vector_store.pkl'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'json'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class EnhancedRAGSystem:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased features
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            max_df=0.9,  # More lenient
            min_df=1,    # Include rare terms
            sublinear_tf=True,  # Better handling of term frequency
            analyzer='word',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        self.document_vectors = None
        self.is_fitted = False
        self.synonym_map = self._build_synonym_map()
        self.keyword_index = defaultdict(set)  # For fast keyword lookup
        
    def _build_synonym_map(self):
        """Build a basic synonym mapping for common terms"""
        synonyms = {
            'work': ['job', 'career', 'employment', 'occupation', 'profession'],
            'study': ['learn', 'education', 'academic', 'school', 'university', 'college'],
            'skill': ['ability', 'talent', 'expertise', 'capability', 'competency'],
            'project': ['task', 'assignment', 'work', 'initiative'],
            'experience': ['background', 'history', 'past', 'previous'],
            'like': ['enjoy', 'love', 'prefer', 'favorite', 'interest'],
            'live': ['reside', 'stay', 'home', 'location', 'place'],
            'family': ['parent', 'mother', 'father', 'sibling', 'brother', 'sister'],
            'hobby': ['interest', 'pastime', 'activity', 'recreation'],
            'travel': ['trip', 'journey', 'visit', 'vacation', 'tour'],
            'technology': ['tech', 'IT', 'computer', 'software', 'programming'],
            'company': ['organization', 'firm', 'business', 'employer'],
            'graduate': ['degree', 'diploma', 'qualification', 'certification'],
            'language': ['speak', 'fluent', 'tongue', 'communication']
        }
        
        # Create bidirectional mapping
        synonym_map = {}
        for main_word, synonyms_list in synonyms.items():
            for synonym in synonyms_list:
                synonym_map[synonym] = main_word
            synonym_map[main_word] = main_word
            
        return synonym_map

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def extract_text_from_file(self, file_path):
        """Extract text from various file types"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()

            elif ext == '.pdf':
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text

            elif ext == '.docx':
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text

            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    return json.dumps(data, indent=2)

            else:
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Expand common contractions
        contractions = {
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text.strip()

    def chunk_text(self, text, chunk_size=400, overlap=100):
        """Enhanced text chunking with better overlap and context preservation"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_size = len(words)
            
            if current_size + sentence_size <= chunk_size:
                current_chunk += sentence + ". "
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunks and overlap > 0:
                    # Take last few words from previous chunk for context
                    prev_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(prev_words) + " " + sentence + ". "
                    current_size = len(prev_words) + sentence_size
                else:
                    current_chunk = sentence + ". "
                    current_size = sentence_size
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def build_keyword_index(self):
        """Build keyword index for fast retrieval"""
        self.keyword_index = defaultdict(set)
        
        for idx, doc in enumerate(self.documents):
            # Extract and normalize keywords
            text = self.preprocess_text(doc['content'])
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
            
            for word in words:
                # Add original word
                self.keyword_index[word].add(idx)
                
                # Add synonym mapping
                if word in self.synonym_map:
                    main_word = self.synonym_map[word]
                    self.keyword_index[main_word].add(idx)

    def fuzzy_keyword_search(self, query, threshold=0.6):
        """Find documents using fuzzy keyword matching"""
        query = self.preprocess_text(query)
        query_words = re.findall(r'\b[a-zA-Z]{2,}\b', query)
        
        candidate_docs = set()
        word_matches = {}
        
        for query_word in query_words:
            # Direct matches
            if query_word in self.keyword_index:
                candidate_docs.update(self.keyword_index[query_word])
                word_matches[query_word] = len(self.keyword_index[query_word])
            
            # Synonym matches
            if query_word in self.synonym_map:
                main_word = self.synonym_map[query_word]
                if main_word in self.keyword_index:
                    candidate_docs.update(self.keyword_index[main_word])
                    word_matches[query_word] = len(self.keyword_index[main_word])
            
            # Fuzzy string matching for similar words
            for indexed_word in self.keyword_index.keys():
                similarity = self.string_similarity(query_word, indexed_word)
                if similarity >= threshold:
                    candidate_docs.update(self.keyword_index[indexed_word])
                    if query_word not in word_matches:
                        word_matches[query_word] = 0
                    word_matches[query_word] += len(self.keyword_index[indexed_word])
        
        return list(candidate_docs), word_matches

    def string_similarity(self, s1, s2):
        """Calculate string similarity using Jaccard similarity on character bigrams"""
        if len(s1) < 2 or len(s2) < 2:
            return 1.0 if s1 == s2 else 0.0
            
        # Create character bigrams
        bigrams1 = set([s1[i:i+2] for i in range(len(s1)-1)])
        bigrams2 = set([s2[i:i+2] for i in range(len(s2)-1)])
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
            
        intersection = bigrams1.intersection(bigrams2)
        union = bigrams1.union(bigrams2)
        
        return len(intersection) / len(union)

    def calculate_relevance_score(self, doc_idx, query, tfidf_score, keyword_matches):
        """Calculate composite relevance score"""
        doc = self.documents[doc_idx]
        doc_text = self.preprocess_text(doc['content'])
        query_text = self.preprocess_text(query)
        
        # TF-IDF score (30% weight)
        tfidf_weight = 0.3 * tfidf_score
        
        # Keyword match score (40% weight)
        query_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', query_text))
        doc_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', doc_text))
        
        keyword_score = 0
        if query_words:
            # Direct matches
            direct_matches = query_words.intersection(doc_words)
            keyword_score += len(direct_matches) / len(query_words) * 0.7
            
            # Synonym matches
            synonym_matches = 0
            for qword in query_words:
                if qword in self.synonym_map:
                    main_word = self.synonym_map[qword]
                    if main_word in doc_words:
                        synonym_matches += 1
            
            keyword_score += (synonym_matches / len(query_words)) * 0.3
        
        keyword_weight = 0.4 * keyword_score
        
        # Length penalty (10% weight) - prefer medium-length chunks
        length_score = min(1.0, len(doc_text.split()) / 200)
        length_weight = 0.1 * length_score
        
        # Phrase proximity score (20% weight)
        proximity_score = self.calculate_phrase_proximity(doc_text, query_text)
        proximity_weight = 0.2 * proximity_score
        
        total_score = tfidf_weight + keyword_weight + length_weight + proximity_weight
        
        return total_score

    def calculate_phrase_proximity(self, doc_text, query_text):
        """Calculate how close query words appear together in document"""
        doc_words = doc_text.split()
        query_words = query_text.split()
        
        if len(query_words) <= 1 or len(doc_words) == 0:
            return 0.0
        
        word_positions = {}
        for i, word in enumerate(doc_words):
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)
        
        min_span = float('inf')
        found_words = 0
        
        for query_word in query_words:
            if query_word in word_positions:
                found_words += 1
                
        if found_words < 2:
            return 0.0
        
        # Find minimum span containing query words
        query_positions = []
        for query_word in query_words:
            if query_word in word_positions:
                query_positions.extend(word_positions[query_word])
        
        if len(query_positions) >= 2:
            query_positions.sort()
            min_span = query_positions[-1] - query_positions[0] + 1
        
        # Convert to proximity score (closer = higher score)
        if min_span == float('inf'):
            return 0.0
        
        proximity_score = 1.0 / (1.0 + min_span / 10.0)
        return min(1.0, proximity_score)

    def process_documents(self, file_paths):
        """Process and store documents with enhanced indexing"""
        self.documents = []

        for file_path in file_paths:
            try:
                text = self.extract_text_from_file(file_path)
                if text:
                    # Preprocess text
                    text = self.preprocess_text(text)
                    
                    # Split into chunks
                    chunks = self.chunk_text(text, chunk_size=400, overlap=100)
                    for chunk in chunks:
                        if len(chunk.split()) >= 10:  # Only keep substantial chunks
                            self.documents.append({
                                'content': chunk,
                                'source': os.path.basename(file_path),
                                'timestamp': datetime.now().isoformat()
                            })
                    logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # Fit vectorizer and create document vectors
        if self.documents:
            document_texts = [doc['content'] for doc in self.documents]
            self.document_vectors = self.vectorizer.fit_transform(document_texts)
            self.is_fitted = True
            
            # Build keyword index for fast retrieval
            self.build_keyword_index()
            
            logger.info(f"Created vectors for {len(self.documents)} document chunks")

        # Save to pickle file
        self.save_vector_store()

    def save_vector_store(self):
        """Save the vector store to disk"""
        try:
            with open(VECTOR_STORE_PATH, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'vectorizer': self.vectorizer,
                    'document_vectors': self.document_vectors,
                    'is_fitted': self.is_fitted,
                    'keyword_index': dict(self.keyword_index)  # Convert defaultdict to dict
                }, f)
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")

    def load_vector_store(self):
        """Load the vector store from disk"""
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                with open(VECTOR_STORE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.vectorizer = data['vectorizer']
                    self.document_vectors = data['document_vectors']
                    self.is_fitted = data['is_fitted']
                    self.keyword_index = defaultdict(set, data.get('keyword_index', {}))
                logger.info("Vector store loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
        return False

    def retrieve_relevant_context(self, query, top_k=5):
        """Enhanced context retrieval with multiple scoring methods"""
        if not self.is_fitted or not self.documents:
            return []

        try:
            # Method 1: TF-IDF similarity
            query_vector = self.vectorizer.transform([self.preprocess_text(query)])
            tfidf_similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Method 2: Fuzzy keyword search
            keyword_candidates, word_matches = self.fuzzy_keyword_search(query)
            
            # Combine candidates from both methods
            all_candidates = set(range(len(self.documents)))  # Consider all documents
            
            # Calculate composite scores for all candidates
            scored_docs = []
            for idx in all_candidates:
                tfidf_score = float(tfidf_similarities[idx])
                
                # Calculate composite relevance score
                relevance_score = self.calculate_relevance_score(
                    idx, query, tfidf_score, word_matches
                )
                
                # Only include documents with some relevance
                if relevance_score > 0.05:  # Much lower threshold
                    scored_docs.append({
                        'index': idx,
                        'score': relevance_score,
                        'tfidf_score': tfidf_score
                    })
            
            # Sort by composite score
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top-k documents
            relevant_docs = []
            for doc_info in scored_docs[:top_k]:
                idx = doc_info['index']
                relevant_docs.append({
                    'content': self.documents[idx]['content'],
                    'source': self.documents[idx]['source'],
                    'similarity': doc_info['score'],
                    'tfidf_similarity': doc_info['tfidf_score']
                })
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query: {query}")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []


# Initialize enhanced RAG system
rag_system = EnhancedRAGSystem()
rag_system.load_vector_store()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(rag_system.documents),
        'vector_store_fitted': rag_system.is_fitted,
        'keyword_index_size': len(rag_system.keyword_index)
    })


@app.route('/upload-documents', methods=['POST'])
@cross_origin(origins='*')
def upload_documents():
    """Upload and process documents about the user"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400

    uploaded_files = []

    for file in files:
        if file and rag_system.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
        else:
            return jsonify({'error': f'Invalid file type: {file.filename}'}), 400

    # Process uploaded documents
    rag_system.process_documents(uploaded_files)

    return jsonify({
        'message': 'Documents uploaded and processed successfully',
        'processed_files': [os.path.basename(f) for f in uploaded_files],
        'total_chunks': len(rag_system.documents)
    })


@app.route('/answerQuestion', methods=['POST'])
@cross_origin(origins='*')
def answer_question():
    """Answer questions about the user using enhanced RAG + OpenAI"""
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        # Retrieve relevant context with enhanced scoring
        relevant_docs = rag_system.retrieve_relevant_context(question, top_k=5)

        if not relevant_docs:
            return jsonify({
                'error': 'No relevant information found. Please upload documents about yourself first.',
                'answer': None,
                'context_used': []
            }), 400

        # Prepare context for GPT
        context = "\n\n".join([
            f"From {doc['source']} (relevance: {doc['similarity']:.3f}): {doc['content']}"
            for doc in relevant_docs
        ])

        # Enhanced system prompt
        system_prompt = """You are Jayant, and your role is to answer questions as if you are speaking about yourself.
        Use the information provided in the context to answer questions naturally and conversationally.

        Instructions:
        - Always respond in the first person ("I", "me", "my")
        - Be conversational and natural, like you are really Jayant
        - Use the context to provide comprehensive answers
        - If you need to make reasonable inferences based on the context, that's okay
        - If the context doesn't have enough information, say: "I don't have enough information about that right now"
        - Don't mention relevance scores or technical details
        - Be helpful and provide as much relevant information as possible from the context

        Remember: You ARE Jayant speaking about yourself, not an AI describing someone else."""

        user_prompt = f"""Context about me:
{context}

Question: {question}

Please answer this question naturally, as if you are Jayant speaking about yourself."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()
        print(answer)
        return jsonify({
            'question': question,
            'answer': answer,
            'context_used': [
                {
                    'source': doc['source'],
                    'relevance_score': doc['similarity'],
                    'tfidf_score': doc['tfidf_similarity']
                } for doc in relevant_docs
            ],
            'total_documents': len(rag_system.documents)
        })

    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/list-documents', methods=['GET'])
def list_documents():
    """List processed documents and their info"""
    document_info = {}
    for doc in rag_system.documents:
        source = doc['source']
        if source not in document_info:
            document_info[source] = {
                'chunks': 0,
                'last_updated': doc['timestamp']
            }
        document_info[source]['chunks'] += 1

    return jsonify({
        'total_chunks': len(rag_system.documents),
        'documents': document_info,
        'vector_store_status': 'fitted' if rag_system.is_fitted else 'not fitted',
        'keyword_index_terms': len(rag_system.keyword_index)
    })


@app.route('/clear-documents', methods=['DELETE'])
def clear_documents():
    """Clear all processed documents"""
    rag_system.documents = []
    rag_system.document_vectors = None
    rag_system.is_fitted = False
    rag_system.keyword_index = defaultdict(set)

    # Remove vector store file
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)

    # Remove uploaded files
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return jsonify({'message': 'All documents cleared successfully'})


@app.route('/debug-search', methods=['POST'])
def debug_search():
    """Debug endpoint to see how search works for a query"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
            
        query = data['query']
        relevant_docs = rag_system.retrieve_relevant_context(query, top_k=10)
        
        return jsonify({
            'query': query,
            'results': relevant_docs,
            'total_candidates': len(relevant_docs)
        })
        
    except Exception as e:
        logger.error(f"Error in debug_search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)