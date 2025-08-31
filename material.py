import os
from src import StudyMaterialRAG
from typing import Optional
from config import config
from logger import logger

def get_rag_system() -> StudyMaterialRAG:
    """Factory function to create RAG system instance with configuration."""
    from src import RAGConfig, StudyMaterialRAG
    rag_config = RAGConfig.from_env()
    return StudyMaterialRAG(rag_config)

def fix_mermaid_syntax(content):
    """
    Fix common Mermaid syntax errors in the generated content.
    """
    import re
    
    # Split content into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    in_mermaid_block = False
    
    for line in lines:
        if '```mermaid' in line:
            in_mermaid_block = True
            fixed_lines.append(line)
        elif '```' in line and in_mermaid_block:
            in_mermaid_block = False
            fixed_lines.append(line)
        elif in_mermaid_block:
            # Fix common syntax errors
            # Replace curly braces with square brackets for node labels
            line = re.sub(r'\{([^}]+)\}', r'[\1]', line)
            
            # Replace parentheses with square brackets for node labels
            line = re.sub(r'\(\s*([^)]+)\s*\)', r'[\1]', line)
            
            # Fix invalid arrow syntax
            line = re.sub(r'\s*PS\s*', ' --> ', line)
            line = re.sub(r'\s*PE\s*', '', line)
            
            # Fix common arrow issues
            line = re.sub(r'->>', '->>', line)  # Ensure proper sequence arrows
            line = re.sub(r'-->>', '-->>', line)
            
            # Remove any remaining invalid syntax
            line = re.sub(r'\b(PS|PE|SQE|DOUBLECIRCLEEND|STADIUMEND|SUBROUTINEEND|PIPE|CYLINDEREND|DIAMOND_STOP|TAGEND|TRAPEND|INVTRAPEND)\b', '', line)
            
            # Fix sequence diagram syntax
            if 'sequenceDiagram' in '\n'.join(fixed_lines[-10:]):  # Check if we're in a sequence diagram
                line = re.sub(r'(\w+)\s*->>\s*(\w+):', r'\1->>\2:', line)
                line = re.sub(r'(\w+)\s*-->\s*(\w+):', r'\1-->\2:', line)
            
            fixed_lines.append(line)
        else:
            # Escape any curly braces in regular content to prevent template variable conflicts
            line = line.replace('{', '{{').replace('}', '}}')
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def validate_mermaid_syntax(content):
    """
    Validate Mermaid syntax and return True if valid, False if errors found.
    """
    import re
    
    lines = content.split('\n')
    in_mermaid_block = False
    errors = []
    
    for i, line in enumerate(lines, 1):
        if '```mermaid' in line:
            in_mermaid_block = True
        elif '```' in line and in_mermaid_block:
            in_mermaid_block = False
        elif in_mermaid_block:
            # Check for common invalid syntax
            if re.search(r'\b(PS|PE|SQE|DOUBLECIRCLEEND)\b', line):
                errors.append(f"Line {i}: Invalid syntax element found")
            if re.search(r'\{[^}]+\}', line):
                errors.append(f"Line {i}: Curly braces found in Mermaid diagram (use square brackets instead)")
            if re.search(r'\(\s*[^)]+\s*\)', line) and not re.search(r'sequenceDiagram', '\n'.join(lines[max(0, i-5):i])):
                errors.append(f"Line {i}: Parentheses used for node labels in flowchart (should be square brackets)")
        else:
            # Check for unescaped template variables in regular content
            template_vars = re.findall(r'\{([^}]+)\}', line)
            if template_vars:
                # Check if these are actual template variables we expect
                expected_vars = ['topic', 'description', 'reference_content', 'css_context', 'teacher_id']
                for var in template_vars:
                    if var not in expected_vars:
                        errors.append(f"Line {i}: Unexpected template variable '{{{var}}}' found")
    
    return len(errors) == 0, errors

def generate_materials(topic: str, teacher_id: str, output_format: str = 'pdf') -> Optional[str]:
    """
    Generate study materials for a given topic.

    Args:
        topic: The topic to generate materials for
        teacher_id: ID of the teacher
        output_format: Output format ('pdf', 'pptx', 'html')

    Returns:
        Path to generated file or None if error
    """
    try:
        # Create RAG system instance
        study_system = get_rag_system()

        materials = study_system.create_full_course_materials(topic, teacher_id)

        # Define upload directory name
        upload_folder = config.UPLOAD_FOLDER
        topic_clean = topic.replace(' ', '_').lower()
        output_format = 'pdf'  # Default format
        # Create 'uploads' directory if it doesn't exist
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Generate markdown file in uploads directory
        output_file = f"{topic_clean}_materials.md"
        file_path = os.path.join(os.getcwd(), upload_folder, output_file)

        with open(file_path, "w", encoding="utf-8") as f:
            # Use configuration for Marp settings
            marp_config = config.MARP_CONFIG
            f.write(f'''---
marp: true
theme: {marp_config["theme"]}
class: {marp_config["class"]}
paginate: {marp_config["paginate"]}
backgroundColor: {marp_config["backgroundColor"]}
color: {marp_config["color"]}
header: {marp_config["header"]}
footer: {marp_config["footer"]}
size: {marp_config["size"]}
width: {marp_config["width"]}
height: {marp_config["height"]}
style: |
  @import url('./assets/dark-theme.css');
  /* Additional Marp-specific overflow prevention */
  section {{
    font-size: 24px;
    line-height: 1.4;
    padding: 40px 60px;
  }}
  @media (max-width: 768px) {{
    section {{
      font-size: 20px;
      padding: 30px 40px;
    }}
  }}
  /* Ensure images don't overflow */
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
  }}
  /* Better text wrapping */
  p, li {{
    margin-bottom: 12px;
  }}
\n''')
            for subtopic, content in materials.items():
                f.write('---\n')
                f.write(f"### {subtopic}\n\n")
                
                # Fix Mermaid syntax errors first
                content = fix_mermaid_syntax(content)
                
                # Validate the fixed content
                is_valid, errors = validate_mermaid_syntax(content)
                if not is_valid:
                    logger.warning(f"Mermaid syntax issues detected in slide '{subtopic}':")
                    for error in errors:
                        logger.warning(f"  - {error}")
                
                # Process content to handle mermaid diagrams and images
                lines = content.split('\n')
                clean_lines = []
                in_mermaid_block = False
                
                for line in lines:
                    # Handle mermaid diagram blocks
                    if '```mermaid' in line:
                        in_mermaid_block = True
                        clean_lines.append(line)
                    elif '```' in line and in_mermaid_block:
                        in_mermaid_block = False
                        clean_lines.append(line)
                    elif in_mermaid_block:
                        # Keep mermaid content as-is (already fixed)
                        clean_lines.append(line)
                    else:
                        # Process regular content
                        clean_line = line.replace("```markdown", "").replace("```", "")
                        
                        # Handle image references - ensure they have proper formatting
                        if '![' in clean_line and '](' in clean_line:
                            # Image reference found, ensure it's properly formatted
                            clean_lines.append(clean_line)
                        else:
                            # Regular text line
                            clean_lines.append(clean_line)
                
                clean_content = '\n'.join(clean_lines)
                clean_content = clean_content.strip()
                
                f.write(clean_content)
                f.write("\n\n")
        os.system(f" mmdc -i {file_path} -o {file_path} ")
        # Generate the PDF, PPTX, or HTML files in 'uploads' directory
        if output_format in ['pptx', 'ppt']:
            pptx_file = f"{topic_clean}_slides.pptx"
            pptx_path = os.path.join(os.getcwd(), upload_folder, pptx_file)
            os.system(f"marp --allow-local-files {file_path} --pptx -o {pptx_path}")
            logger.info(f"PPTX file generated at: {pptx_path}")
            return pptx_path

        elif output_format == 'pdf':
            pdf_file = f"{topic_clean}_materials.pdf"
            pdf_path = os.path.join(os.getcwd(), upload_folder, pdf_file)
            os.system(f"marp --allow-local-files {file_path} --pdf -o {pdf_path}")
            logger.info(f"PDF file generated at: {pdf_path}")
            return pdf_path

        elif output_format == 'html':
            html_file = f"{topic_clean}.html"
            html_path = os.path.join(os.getcwd(), upload_folder, html_file)
            os.system(f"marp --allow-local-files {file_path} --html -o {html_path}")
            logger.info(f"HTML file generated at: {html_path}")
            return html_path

        else:
            logger.warning(f"Unsupported output format: {output_format}")
            return None

    except Exception as e:
        logger.error(f"Error occurred in generate_materials: {str(e)}")
        return None
