from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from pdf2image import convert_from_path
import pytesseract
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_core.embeddings import Embeddings
import re
import requests
import json
import spacy
from datetime import datetime
from typing import List
from uuid import uuid4  # Add this import at the top of the file
from langchain_core.documents import Document  # Add this import at the top of the file
import time
from tenacity import retry, stop_after_attempt, wait_fixed
from sentence_transformers import SentenceTransformer
import numpy as np

# Load SpaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# REBEL model for relationship extraction
rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    length_function=len,
    is_separator_regex=False,
)

# Remove the GroqEmbeddings class and use SentenceTransformer directly
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode

class PaddedSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-mpnet-base-v2', target_dim=1536):
        self.model = SentenceTransformer(model_name)
        self.target_dim = target_dim

    def pad_embedding(self, embedding):
        if len(embedding) >= self.target_dim:
            return embedding[:self.target_dim]
        else:
            return np.pad(embedding, (0, self.target_dim - len(embedding)), 'constant').tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return [self.pad_embedding(emb) for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return self.pad_embedding(embedding)

# Initialize embeddings
embeddings = PaddedSentenceTransformerEmbeddings()

vector_store = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    index_name="document_index",
    node_label="Document",
    text_node_property="text",
    embedding_node_property="embedding",
)

def clean_entity(entity):
    cleaned = re.sub(r'[^\w\s]', '', entity)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace(" ", "").replace(" ", "").replace("<pad>", "")
    print(f"Processing text: {text_replaced}")  # Debug print
    for token in text_replaced.split():
        if token == "<triplet>":
            if relation and subject and object_:
                clean_subj = clean_entity(subject)
                clean_obj = clean_entity(object_)
                if clean_subj and clean_obj and clean_subj != clean_obj:
                    relations.append({
                        'head': clean_subj,
                        'type': relation.strip().replace(" ", "_"),
                        'tail': clean_obj
                    })
            relation, subject, object_ = '', '', ''
            current = 't'
        elif token == "<subj>":
            current = 's'
        elif token == "<obj>":
            current = 'o'
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    # Add the last relation if it exists
    if relation and subject and object_:
        clean_subj = clean_entity(subject)
        clean_obj = clean_entity(object_)
        if clean_subj and clean_obj and clean_subj != clean_obj:
            relations.append({
                'head': clean_subj,
                'type': relation.strip().replace(" ", "_"),
                'tail': clean_obj
            })
    
    return relations

def extract_relationships_with_context(text):
    # Use a more robust NER and relation extraction method
    doc = nlp(text)
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                subject = token.subtree
                verb = token.head
                for child in verb.children:
                    if child.dep_ in ["dobj", "attr", "prep"]:
                        object = child.subtree
                        relationships.append({
                            'head': ' '.join([t.text for t in subject]).strip(),
                            'type': verb.lemma_,
                            'tail': ' '.join([t.text for t in object]).strip()
                        })
    return relationships

def extract_text_from_pdf(pdf_path):
    try:
        pages = convert_from_path(pdf_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        if text:
            print(f"Extracted text (first 500 characters): {text[:500]}")
        else:
            print("No text extracted from PDF.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_metadata(text):
    doc = nlp(text)
    metadata = {
        "dates": [],
        "organizations": [],
        "locations": [],
        "people": []
    }
    for ent in doc.ents:
        if ent.label_ == "DATE":
            metadata["dates"].append(ent.text)
        elif ent.label_ == "ORG":
            metadata["organizations"].append(ent.text)
        elif ent.label_ == "GPE":
            metadata["locations"].append(ent.text)
        elif ent.label_ == "PERSON":
            metadata["people"].append(ent.text)
    return metadata

def process_pdf_to_kg(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text extracted from PDF. Aborting process.")
        return [], {}
    
    chunks = text_splitter.split_text(text)
    all_relationships = []
    metadata = {}

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            for i, chunk in enumerate(chunks):
                relationships = extract_relationships_with_context(chunk)
                
                chunk_node_query = (
                    "CREATE (c:Chunk {id: $chunk_id, text: $text}) "
                    "RETURN id(c) as node_id"
                )
                chunk_node = session.run(chunk_node_query, chunk_id=i, text=chunk).single()['node_id']
                
                for rel in relationships:
                    relationship_query = (
                        "MERGE (h:Entity {name: $head}) "
                        "MERGE (t:Entity {name: $tail}) "
                        "MERGE (h)-[r:RELATIONSHIP {type: $type}]->(t) "
                        "MERGE (h)-[:MENTIONED_IN]->(c:Chunk {id: $chunk_id}) "
                        "MERGE (t)-[:MENTIONED_IN]->(c)"
                    )
                    session.run(relationship_query, 
                                head=rel['head'], 
                                tail=rel['tail'], 
                                type=rel['type'], 
                                chunk_id=i)
                
                all_relationships.extend(relationships)
                
                vector_store.add_texts(
                    texts=[chunk],
                    metadatas=[{'chunk_id': i}],
                    ids=[str(chunk_node)]
                )

    return all_relationships, metadata

def query_knowledge_graph(question):
    vector_results = vector_store.similarity_search_with_score(question, k=5)
    
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            query = (
                "MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity) "
                "WHERE e1.name CONTAINS $question OR e2.name CONTAINS $question OR r.type CONTAINS $question "
                "WITH e1, r, e2 "
                "MATCH (e1)-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(e2) "
                "RETURN e1.name AS source, r.type AS relationship, e2.name AS target, c.text AS context "
                "LIMIT 5"
            )
            result = session.run(query, question=question)
            kg_relationships = [dict(record) for record in result]

    return kg_relationships, vector_results

def generate_response(question, kg_relationships, vector_results):
    context = "\n".join([f"{rel['source']} {rel['relationship']} {rel['target']}: {rel['context']}" for rel in kg_relationships])
    context += "\n" + "\n".join([f"Relevant text: {doc.page_content}" for doc, _ in vector_results])

    prompt = f"""Question: {question}

Context:
{context}

Based solely on the above context, please provide a concise answer to the question. If the information is not sufficient to answer the question, please state that there is not enough information to provide an answer."""

    # Use Groq API for response generation
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error in generating response: {response.status_code}"

# ... (keep the rest of the file as is)

if __name__ == "__main__":
    pdf_path = "shimadzu.pdf"
    process_pdf_to_kg(pdf_path)