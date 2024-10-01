import streamlit as st
import tempfile
import os
from main import process_pdf_to_kg, extract_text_from_pdf, query_knowledge_graph, generate_response
import networkx as nx
from pyvis.network import Network

st.set_page_config(page_title="Knowledge Graph RAG System", layout="wide")
st.title("Knowledge Graph RAG System")

if 'kg_built' not in st.session_state:
    st.session_state.kg_built = False

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(tmp_file_path)
        if text:
            st.success("Text extracted successfully.")
            st.text_area("Extracted Text (first 500 characters)", text[:500], height=200)
        else:
            st.error("Failed to extract text from the PDF.")

    if st.button("Convert to Knowledge Graph"):
        with st.spinner("Converting PDF to Knowledge Graph..."):
            all_relationships, metadata = process_pdf_to_kg(tmp_file_path)

        if all_relationships:
            num_relationships = len(all_relationships)
            st.success(f"Total relationships extracted: {num_relationships}")
            
            with st.expander("Sample of relationships extracted"):
                for rel in all_relationships[:10]:
                    st.write(f"({rel['head']}) -[{rel['type']}]-> ({rel['tail']})")
                if len(all_relationships) > 10:
                    st.write("... (more relationships not shown)")

            st.subheader("Knowledge Graph Visualization")
            G = nx.DiGraph()
            for rel in all_relationships:
                G.add_edge(rel['head'], rel['tail'], label=rel['type'])

            net = Network(notebook=True, width="100%", height="500px", directed=True)
            for node in G.nodes():
                net.add_node(node)
            for edge in G.edges(data=True):
                net.add_edge(edge[0], edge[1], title=edge[2]['label'])

            net.save_graph("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                graph_html = f.read()
            st.components.v1.html(graph_html, height=600)

        else:
            st.warning("No relationships extracted.")
        
        st.session_state.kg_built = True
        st.success("Conversion complete! Knowledge Graph built successfully.")

    os.unlink(tmp_file_path)

if st.session_state.kg_built:
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        with st.spinner("Processing your question..."):
            kg_relationships, vector_results = query_knowledge_graph(user_question)
        
        with st.expander("Knowledge Graph Relationships"):
            if kg_relationships:
                for rel in kg_relationships:
                    st.write(f"- {rel['source']} {rel['relationship']} {rel['target']}")
            else:
                st.write("No direct relationships found in the knowledge graph.")
        
        with st.expander("Vector Search Results"):
            if vector_results:
                for doc, score in vector_results:
                    st.write(f"- {doc.page_content[:100]}... (Similarity: {score:.4f})")
            else:
                st.write("No similar documents found.")
        
        if kg_relationships or vector_results:
            response = generate_response(user_question, kg_relationships, vector_results)
            st.subheader("Generated Answer")
            st.write(response)
        else:
            st.warning("Sorry, I couldn't find any relevant information to answer your question.")

st.info("Note: Make sure your Neo4j database is running and properly configured in the .env file.")