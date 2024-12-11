import os
import PyPDF2
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import time
import torch
from transformers import AutoTokenizer, AutoModel

class RAGChatbot:
    def __init__(self):
        # Initialize API keys from environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.pinecone_api_key or not self.groq_api_key:
            raise ValueError("Please set PINECONE_API_KEY and GROQ_API_KEY environment variables")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize embedding model
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-small-v2')
        
        # Pinecone index configuration
        self.cloud = os.getenv('PINECONE_CLOUD', 'aws')
        self.region = os.getenv('PINECONE_REGION', 'us-east-1')
        self.index_name = 'cv-rag-index'
        self.dimension = 384  # Dimension for e5-small-v2
        
        # Create or connect to Pinecone index
        self.setup_pinecone_index()
    
    def setup_pinecone_index(self):
        """Create Pinecone index if it doesn't exist"""
        spec = ServerlessSpec(cloud=self.cloud, region=self.region)
        
        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create index
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=spec
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    
    def generate_embeddings(self, text):
        """Generate embeddings using Hugging Face model"""
        # Prefix the text with "query: " or "passage: " depending on use
        text = f"passage: {text}"
        
        # Tokenize the text
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**tokens)
        
        # Mean pooling
        embeddings = self.mean_pooling(model_output, tokens['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].tolist()
    
    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on model output"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def index_document(self, pdf_path):
        """Index a PDF document in Pinecone"""
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Index each chunk
        for i, chunk in enumerate(chunks):
            # Generate embeddings
            embedding = self.generate_embeddings(chunk)
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[
                (f"{pdf_path}_chunk_{i}", embedding, {"text": chunk})
            ])
        
        print(f"Indexed document: {pdf_path}")
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant context using semantic search"""
        # Generate query embedding
        query_embedding = self.generate_embeddings(query)
        
        # Perform similarity search
        results = self.index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        # Extract and return relevant contexts
        contexts = [
            match['metadata']['text'] for match in results['matches']
        ]
        return contexts
    
    def generate_response(self, query):
        """Generate a response using retrieved context"""
        # Retrieve relevant context
        contexts = self.retrieve_relevant_context(query)
        
        # Prepare prompt with context
        prompt = f"""Contexto: {' '.join(contexts)}
        
Pregunta: {query}

Basado en el contexto proporcionado, por favor responde a la pregunta de manera completa y precisa:"""
        
        # Generate response using Groq
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un asistente útil que utiliza el contexto para responder preguntas."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        
        return chat_completion.choices[0].message.content
    
    def chat(self):
        """Interactive chat interface"""
        print("RAG Chatbot Initialized. Type 'exit' to quit.")
        while True:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                break
            
            try:
                response = self.generate_response(query)
                print("\nChatbot:", response)
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    # Initialize the chatbot
    chatbot = RAGChatbot()
    
    # Index a PDF document (replace with your CV path)
    chatbot.index_document('C:/Users/Lean/Downloads/cv_tomas.pdf')
    
    # Start interactive chat
    chatbot.chat()

if __name__ == "__main__":
    main()

# Requirements:
# pip install pinecone-client groq PyPDF2 torch transformers
# Set environment variables:
# export PINECONE_API_KEY='your_pinecone_key'
# export GROQ_API_KEY='your_groq_key'