import streamlit as st
import os
from src import StudyMaterialRAG, RAGConfig
from material import generate_materials
from logger import logger
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

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

def analyze_document_content(text):
    """Analyze document content for advanced metrics."""
    # Basic text statistics
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))
    paragraph_count = len(text.split('\n\n'))
    
    # Readability metrics
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    avg_sentences_per_paragraph = sentence_count / paragraph_count if paragraph_count > 0 else 0
    
    # Content analysis
    technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    code_snippets = len(re.findall(r'```.*?```', text, re.DOTALL))
    mermaid_diagrams = len(re.findall(r'```mermaid.*?```', text, re.DOTALL))
    
    # Topic diversity (simple keyword analysis)
    common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in common_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    top_topics = word_freq.most_common(10)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_sentences_per_paragraph': avg_sentences_per_paragraph,
        'technical_terms': technical_terms,
        'code_snippets': code_snippets,
        'mermaid_diagrams': mermaid_diagrams,
        'top_topics': top_topics
    }

def generate_content_quality_report(analysis_results):
    """Generate a comprehensive content quality report."""
    report = f"""
# 📊 Content Quality Analysis Report

## 📈 Basic Statistics
- **Total Words**: {analysis_results['word_count']:,}
- **Total Sentences**: {analysis_results['sentence_count']:,}
- **Total Paragraphs**: {analysis_results['paragraph_count']:,}

## 📚 Content Structure
- **Average Words per Sentence**: {analysis_results['avg_words_per_sentence']:.1f}
- **Average Sentences per Paragraph**: {analysis_results['avg_sentences_per_paragraph']:.1f}

## 🔧 Technical Content
- **Technical Terms Identified**: {analysis_results['technical_terms']:,}
- **Code Snippets**: {analysis_results['code_snippets']:,}
- **Mermaid Diagrams**: {analysis_results['mermaid_diagrams']:,}

## 🎯 Top Topics
"""
    
    for i, (topic, freq) in enumerate(analysis_results['top_topics'][:10], 1):
        report += f"{i}. **{topic.title()}**: {freq} occurrences\n"
    
    return report

# --- Main App ---
initialize_rag_system()

# Create tabs
tab1, tab2 = st.tabs(["📝 Generate Materials", "🔬 Advanced Analysis"])

with tab1:
    st.header("Generate Study Materials")
    
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

with tab2:
    st.header("🔬 Advanced Document Analysis")
    st.markdown("Analyze document content quality, structure, and technical depth based on research methodologies.")
    
    # Document analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Analyze Syllabus")
        syllabus_analysis_file = st.file_uploader("Upload Syllabus for Analysis", type="pdf", key="analysis_syllabus")
        if st.button("Analyze Syllabus") and syllabus_analysis_file:
            with st.spinner("Analyzing syllabus content..."):
                try:
                    rag = st.session_state.rag_system
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(syllabus_analysis_file.getvalue())
                        tmp_path = tmp.name
                    
                    syllabus_text = rag.extract_text_from_pdf(tmp_path)
                    analysis_results = analyze_document_content(syllabus_text)
                    
                    # Display results
                    st.success("Syllabus analysis complete!")
                    
                    # Metrics in columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Total Words", f"{analysis_results['word_count']:,}")
                        st.metric("Technical Terms", analysis_results['technical_terms'])
                    with metric_col2:
                        st.metric("Sentences", analysis_results['sentence_count'])
                        st.metric("Code Snippets", analysis_results['code_snippets'])
                    with metric_col3:
                        st.metric("Paragraphs", analysis_results['paragraph_count'])
                        st.metric("Mermaid Diagrams", analysis_results['mermaid_diagrams'])
                    
                    # Quality report
                    st.subheader("📊 Quality Report")
                    report = generate_content_quality_report(analysis_results)
                    st.markdown(report)
                    
                    # Topic visualization
                    st.subheader("🎯 Topic Distribution")
                    if analysis_results['top_topics']:
                        topics_df = pd.DataFrame(analysis_results['top_topics'], columns=['Topic', 'Frequency'])
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=topics_df.head(10), x='Frequency', y='Topic', ax=ax)
                        plt.title('Top 10 Topics by Frequency')
                        st.pyplot(fig)
                    
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    with col2:
        st.subheader("📚 Analyze Reference")
        reference_analysis_file = st.file_uploader("Upload Reference for Analysis", type="pdf", key="analysis_reference")
        if st.button("Analyze Reference") and reference_analysis_file:
            with st.spinner("Analyzing reference content..."):
                try:
                    rag = st.session_state.rag_system
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(reference_analysis_file.getvalue())
                        tmp_path = tmp.name
                    
                    reference_text = rag.extract_text_from_pdf(tmp_path)
                    analysis_results = analyze_document_content(reference_text)
                    
                    # Display results
                    st.success("Reference analysis complete!")
                    
                    # Metrics in columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Total Words", f"{analysis_results['word_count']:,}")
                        st.metric("Technical Terms", analysis_results['technical_terms'])
                    with metric_col2:
                        st.metric("Sentences", analysis_results['sentence_count'])
                        st.metric("Code Snippets", analysis_results['code_snippets'])
                    with metric_col3:
                        st.metric("Paragraphs", analysis_results['paragraph_count'])
                        st.metric("Mermaid Diagrams", analysis_results['mermaid_diagrams'])
                    
                    # Quality report
                    st.subheader("📊 Quality Report")
                    report = generate_content_quality_report(analysis_results)
                    st.markdown(report)
                    
                    # Topic visualization
                    st.subheader("🎯 Topic Distribution")
                    if analysis_results['top_topics']:
                        topics_df = pd.DataFrame(analysis_results['top_topics'], columns=['Topic', 'Frequency'])
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=topics_df.head(10), x='Frequency', y='Topic', ax=ax)
                        plt.title('Top 10 Topics by Frequency')
                        st.pyplot(fig)
                    
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    # Comparative analysis section
    st.header("⚖️ Comparative Analysis")
    st.markdown("Upload both documents to compare their content quality and structure.")
    
    comp_syllabus = st.file_uploader("Syllabus for Comparison", type="pdf", key="comp_syllabus")
    comp_reference = st.file_uploader("Reference for Comparison", type="pdf", key="comp_reference")
    
    if st.button("Compare Documents") and comp_syllabus and comp_reference:
        with st.spinner("Performing comparative analysis..."):
            try:
                rag = st.session_state.rag_system
                
                # Analyze both documents
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp1:
                    tmp1.write(comp_syllabus.getvalue())
                    syllabus_text = rag.extract_text_from_pdf(tmp1.name)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                    tmp2.write(comp_reference.getvalue())
                    reference_text = rag.extract_text_from_pdf(tmp2.name)
                
                syllabus_analysis = analyze_document_content(syllabus_text)
                reference_analysis = analyze_document_content(reference_text)
                
                # Create comparison table
                comparison_data = {
                    'Metric': ['Word Count', 'Sentence Count', 'Paragraph Count', 'Technical Terms', 'Code Snippets', 'Mermaid Diagrams'],
                    'Syllabus': [
                        syllabus_analysis['word_count'],
                        syllabus_analysis['sentence_count'], 
                        syllabus_analysis['paragraph_count'],
                        syllabus_analysis['technical_terms'],
                        syllabus_analysis['code_snippets'],
                        syllabus_analysis['mermaid_diagrams']
                    ],
                    'Reference': [
                        reference_analysis['word_count'],
                        reference_analysis['sentence_count'],
                        reference_analysis['paragraph_count'], 
                        reference_analysis['technical_terms'],
                        reference_analysis['code_snippets'],
                        reference_analysis['mermaid_diagrams']
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar chart comparison
                metrics = comparison_data['Metric']
                syllabus_vals = comparison_data['Syllabus']
                reference_vals = comparison_data['Reference']
                
                x = range(len(metrics))
                ax1.bar([i - 0.2 for i in x], syllabus_vals, 0.4, label='Syllabus', alpha=0.8)
                ax1.bar([i + 0.2 for i in x], reference_vals, 0.4, label='Reference', alpha=0.8)
                ax1.set_xticks(x)
                ax1.set_xticklabels(metrics, rotation=45, ha='right')
                ax1.set_title('Document Comparison')
                ax1.legend()
                
                # Topic overlap analysis
                syllabus_topics = set([topic for topic, _ in syllabus_analysis['top_topics'][:20]])
                reference_topics = set([topic for topic, _ in reference_analysis['top_topics'][:20]])
                overlap = len(syllabus_topics.intersection(reference_topics))
                only_syllabus = len(syllabus_topics - reference_topics)
                only_reference = len(reference_topics - syllabus_topics)
                
                ax2.pie([overlap, only_syllabus, only_reference], 
                       labels=['Shared Topics', 'Syllabus Only', 'Reference Only'],
                       autopct='%1.1f%%')
                ax2.set_title('Topic Overlap Analysis')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Cleanup
                os.unlink(tmp1.name)
                os.unlink(tmp2.name)
                
                st.success("Comparative analysis complete!")
                
            except Exception as e:
                st.error(f"Comparative analysis failed: {e}")

st.info("💡 Tip: Use the Advanced Analysis tab to evaluate document quality and structure before generating materials.")
