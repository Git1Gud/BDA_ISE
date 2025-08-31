import streamlit as st
import os
from src import StudyMaterialRAG, RAGConfig
from material import generate_materials
from logger import logger
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Study Material Generator",
    page_icon="📚",
    layout="wide",
)

# --- Title and Description ---
st.title("📚 AI-Powered Study Material Generator")
st.markdown("Upload a syllabus and reference PDF, enter a topic or unit, and get your customized study materials.")

# --- Session State Initialization ---
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# --- Helper Functions ---
def initialize_rag_system():
    """Initializes the RAG system and stores it in the session state."""
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG System... This may take a moment."):
            try:
                from src.utils import EnvironmentManager
                EnvironmentManager.load_environment()
                config = RAGConfig.from_env()
                st.session_state.rag_system = StudyMaterialRAG(config)
                logger.info("RAG system initialized successfully for Streamlit app.")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                logger.error(f"Streamlit RAG initialization failed: {e}")
                st.stop()

# --- UI Components ---
initialize_rag_system()

with st.sidebar:
    st.header("1. Process Syllabus")
    syllabus_file = st.file_uploader("Upload Syllabus PDF", type="pdf", key="syllabus")
    if st.button("Process Syllabus"):
        if syllabus_file and st.session_state.rag_system:
            with st.spinner("Processing syllabus..."):
                try:
                    rag = st.session_state.rag_system
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(syllabus_file.getvalue())
                        tmp_path = tmp.name
                    
                    syllabus_text = rag.extract_text_from_pdf(tmp_path)
                    rag.add_syllabus_content(syllabus_text, {"source": syllabus_file.name})
                    st.success("Syllabus processed successfully!")
                    os.unlink(tmp_path) # Clean up temp file
                except Exception as e:
                    st.error(f"Syllabus processing failed: {e}")
        elif not syllabus_file:
            st.warning("Please upload a syllabus PDF first.")

    st.header("2. Process Reference")
    reference_file = st.file_uploader("Upload Reference PDF", type="pdf", key="reference")
    if st.button("Process Reference"):
        if reference_file and st.session_state.rag_system:
            with st.spinner("Processing reference..."):
                try:
                    rag = st.session_state.rag_system
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(reference_file.getvalue())
                        tmp_path = tmp.name

                    reference_text = rag.extract_text_from_pdf(tmp_path)
                    rag.add_reference_content(reference_text, {"source": reference_file.name})
                    st.success("Reference processed successfully!")
                    os.unlink(tmp_path) # Clean up temp file
                except Exception as e:
                    st.error(f"Reference processing failed: {e}")
        elif not reference_file:
            st.warning("Please upload a reference PDF first.")

st.header("3. Generate Materials")
query = st.text_input("Enter your query", placeholder="e.g., 'unit 1.2' or 'Middleware Concepts'", label_visibility="hidden")
if st.button("Generate Materials", type="primary"):
    if query and st.session_state.rag_system:
        with st.spinner("Generating materials..."):
            try:
                rag = st.session_state.rag_system
                output_path = generate_materials(rag, query, teacher_id="StreamlitUser", output_format='pdf')

                if output_path and os.path.exists(output_path):
                    st.success("Study materials generated successfully!")
                    
                    pdf_path = output_path.replace('.md', '.pdf')
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download Generated PDF",
                                data=pdf_file,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf"
                            )
                    
                    with open(output_path, "r", encoding="utf-8") as md_file:
                        st.download_button(
                            label="Download Generated Markdown",
                            data=md_file,
                            file_name=os.path.basename(output_path),
                            mime="text/markdown"
                        )
                else:
                    st.error("Failed to generate study materials. Check logs.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif not query:
        st.warning("Please enter a query.")

