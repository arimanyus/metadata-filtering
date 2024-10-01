from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import torch
import requests
import json

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH, database=NEO4J_DATABASE)

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    length_function=len,
    is_separator_regex=False,
)

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

def clean_entity(entity):
    cleaned = re.sub(r'[^\w\s]', '', entity)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def preprocess_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters except periods and commas
    text = re.sub(r'[^\w\s.,]', '', text)
    return text

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    print(f"Processed text for relation extraction: {text_replaced}")  # Debug print
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                clean_subj = clean_entity(subject)
                clean_obj = clean_entity(object_)
                if clean_subj and clean_obj and clean_subj != clean_obj:
                    relations.append({
                        'head': clean_subj,
                        'type': relation.strip().replace(" ", "_"),
                        'tail': clean_obj
                    })
                relation = ''
                subject = ''
                object_ = ''
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
        print(f"Current token: {token}, Current: {current}, Subject: {subject}, Object: {object_}, Relation: {relation}")  # Debug print
    
    # Add the last relation if it exists
    if subject != '' and relation != '' and object_ != '':
        clean_subj = clean_entity(subject)
        clean_obj = clean_entity(object_)
        if clean_subj and clean_obj and clean_subj != clean_obj:
            relations.append({
                'head': clean_subj,
                'type': relation.strip().replace(" ", "_"),
                'tail': clean_obj
            })
    
    print(f"Extracted relations: {relations}")
    return relations

def extract_fallback_relationships(text):
    sentences = text.split('.')
    relationships = []
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) > 2:
            relationships.append({
                'head': words[0],
                'type': 'RELATED_TO',
                'tail': words[-1]
            })
    return relationships

def extract_relationships_with_context(text):
    print(f"Processing chunk: {text[:100]}...")
    text = preprocess_text(text)
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": 5
    }
    generated_tokens = model.generate(**model_inputs, **gen_kwargs)
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    relationships = []
    for sentence_pred in decoded_preds:
        print(f"Model output: {sentence_pred}")
        extracted_relations = extract_relations_from_model_output(sentence_pred)
        relationships.extend(extracted_relations)
    if not relationships:
        print("No relationships extracted by the model. Using fallback method.")
        relationships = []
    print(f"Extracted relationships: {relationships}")
    return relationships

def extract_relationships_gpt(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Extract relationships from the following text. Format the output as a list of JSON objects, each containing 'head', 'type', and 'tail' keys. Do not put hyphens in the relationship type.

Text: {text}

Example output format:
[
    {{"head": "Entity1", "type": "RELATION", "tail": "Entity2"}},
    {{"head": "Entity3", "type": "ANOTHER_RELATION", "tail": "Entity4"}}
]
"""

    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        try:
            relationships = json.loads(content)
            print(f"GPT extracted relationships: {relationships}")
            return relationships
        except json.JSONDecodeError:
            print("Failed to parse GPT output as JSON")
            print(f"Raw GPT output: {content}")
            return []
    else:
        print(f"Error in GPT API call: {response.status_code}")
        return []

def clear_database():
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

def insert_relationships(relationships):
    with driver.session(database=NEO4J_DATABASE) as session:
        for relation in relationships:
            if isinstance(relation, dict) and 'head' in relation and 'tail' in relation:
                head = relation.get('head', '')
                tail = relation.get('tail', '')
                rel = relation.get('type', 'RELATED_TO').upper()
            elif isinstance(relation, dict) and 'subject' in relation and 'object' in relation:
                head = relation.get('subject', '')
                tail = relation.get('object', '')
                rel = relation.get('predicate', 'RELATED_TO').upper()
            else:
                print(f"Skipping invalid relationship format: {relation}")
                continue

            if head and tail:
                rel = re.sub(r'[^\w]+', '_', rel)
                rel = re.sub(r'^_|_$', '', rel)
                if not rel:
                    rel = 'RELATED_TO'
                print(f"Creating relationship: ({head}) -[{rel}]-> ({tail})")
                query = (
                    "MERGE (h:Entity {name: $head_name}) "
                    "MERGE (t:Entity {name: $tail_name}) "
                    f"MERGE (h)-[r:{rel}]->(t)"
                )
                session.run(query, head_name=head, tail_name=tail)
            else:
                print(f"Skipping relationship due to missing head or tail: {relation}")
#ensure we're not adding any duplicate relationships that aren;t actually required here
def process_pdf_to_kg(pdf_path, additional_context="", use_gpt=False):
    try:
        clear_database()
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print("No text extracted from PDF. Aborting process.")
            return []
        
        all_relationships = []

        if additional_context:
            print("Processing additional context...")
            if use_gpt:
                context_relationships = extract_relationships_gpt(additional_context)
            else:
                context_relationships = extract_relationships_with_context(additional_context)
            
            all_relationships.extend(context_relationships)
            print(f"Relationships extracted from context: {len(context_relationships)}")

        chunks = text_splitter.split_text(text)
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            if use_gpt:
                relationships = extract_relationships_gpt(chunk)
            else:
                relationships = extract_relationships_with_context(chunk)
            all_relationships.extend(relationships)
            print(f"Extracted relationships from chunk {i+1}: {relationships}")
        
        print("All Relationships:", all_relationships)
        insert_relationships(all_relationships)
        return all_relationships
    finally:
        driver.close()

if __name__ == "__main__":
    pdf_path = "shimadzu.pdf"
    process_pdf_to_kg(pdf_path)