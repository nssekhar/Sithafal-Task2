import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class DataIngestion:
    def __init__(self, urls):
        self.urls = urls
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.metadata = []

    def crawl_and_scrape(self):
        for url in self.urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            self.process_content(text, url)

    def process_content(self, text, url):
        chunks = self.segment_content(text)
        for chunk in chunks:
            embedding = self.model.encode(chunk)
            self.embeddings.append(embedding)
            self.metadata.append(url)

    def segment_content(self, text):
        return text.split('\n\n')  

    def store_embeddings(self):
        embedding_matrix = np.array(self.embeddings).astype('float32')
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        faiss.write_index(index, 'embeddings.index')
class QueryHandler:
    def __init__(self, index, model):
        self.index = index
        self.model = model

    def handle_query(self, query):
        query_embedding = self.model.encode(query)
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k=5)  
        return I  
    from transformers import pipeline

class ResponseGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  
        self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
        self.pipe = pipeline('text-generation', model=self.llm, tokenizer=self.tokenizer)



    def generate_response(self, relevant_chunks, user_query):
        context = " ".join(relevant_chunks)
        prompt = f"Context: {context}\nQuestion: {user_query}\nAnswer:"
        
        response = self.pipe(prompt, max_length=150)  
        return response[0]['generated_text']
if __name__ == "__main__":
    urls = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]

    
    ingestion = DataIngestion(urls)
    ingestion.crawl_and_scrape()
    ingestion.store_embeddings()

    
    index = faiss.read_index('embeddings.index')

    query_handler = QueryHandler(index, ingestion.model)
    response_generator = ResponseGenerator()

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        relevant_indices = query_handler.handle_query(user_query)
        relevant_chunks = [ingestion.metadata[i] for i in relevant_indices[0]]
        response = response_generator.generate_response(relevant_chunks, user_query)

        print(response)
