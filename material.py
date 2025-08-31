# from flask import request, jsonify, send_file
import os
from RAG import StudyMaterialRAG

study_system=StudyMaterialRAG()

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

def generate_materials(topic,teacher_id):
    try:
        # data = request.get_json()
        # data['teacher_id']=1
        
        # Check if required data is provided
        # if not data or 'topic' not in data or 'teacher_id' not in data:
        #     print("Error: Missing required fields 'topic' or 'teacher_id' in the request.")
        #     return jsonify({"error": "Topic and teacher_id are required"}), 400
        
        # topic = data['topic']
        # teacher_id = data['teacher_id']
        # output_format = data.get('output_format', 'pptx')  # Default to PPTX
   
        materials = study_system.create_full_course_materials(topic, teacher_id)

        # Define upload directory name
        upload_folder = "uploads"
        topic='DC'
        output_format='pdf'
        # Create 'uploads' directory if it doesn't exist
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Generate markdown file in uploads directory
        output_file = f"{topic.replace(' ', '_').lower()}_materials.md"
        file_path = os.path.join(os.getcwd(), upload_folder, output_file)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write('''---
marp: true
theme: uncover
class: invert
paginate: true
backgroundColor: #1a1a1a
color: #ffffff
header: 'Study Materials'
footer: 'Generated by AI Assistant'
size: 16:9
width: 1280
height: 720
style: |
  @import url('./assets/dark-theme.css');
  /* Additional Marp-specific overflow prevention */
  section {
    font-size: 24px;
    line-height: 1.4;
    padding: 40px 60px;
  }
  @media (max-width: 768px) {
    section {
      font-size: 20px;
      padding: 30px 40px;
    }
  }
  /* Ensure images don't overflow */
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
  }
  /* Better text wrapping */
  p, li {
    margin-bottom: 12px;
  }
\n''')
            for subtopic, content in materials.items():
                f.write('---\n')
                f.write(f"### {subtopic}\n\n")
                
                # Fix Mermaid syntax errors first
                content = fix_mermaid_syntax(content)
                
                # Validate the fixed content
                is_valid, errors = validate_mermaid_syntax(content)
                if not is_valid:
                    print(f"Warning: Mermaid syntax issues detected in slide '{subtopic}':")
                    for error in errors:
                        print(f"  - {error}")
                
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
            pptx_file = f"{topic.replace(' ', '_').lower()}_slides.pptx"
            pptx_path = os.path.join(os.getcwd(), upload_folder, pptx_file)
            os.system(f"marp --allow-local-files {file_path} --pptx -o {pptx_path}")
            print(f"PPTX file generated at: {pptx_path}")
            # response = send_file(
            #     pptx_path,
            #     as_attachment=True,
            #     download_name=pptx_file,
            #     mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            # )
            # return response

        elif output_format == 'pdf':
            pdf_file = f"{topic.replace(' ', '_').lower()}_materials.pdf"
            pdf_path = os.path.join(os.getcwd(), upload_folder, pdf_file)
            os.system(f"marp --allow-local-files {file_path} --pdf -o {pdf_path}")
            print(f"PDF file generated at: {pdf_path}")
            # response = send_file(
            #     pdf_path,
            #     as_attachment=True,
            #     download_name=pdf_file,
            #     mimetype="application/pdf"
            # )
            # return response

        elif output_format == 'html':
            html_file = f"{topic.replace(' ', '_').lower()}.html"
            html_path = os.path.join(os.getcwd(), upload_folder, html_file)
            os.system(f"marp --allow-local-files {file_path} --html -o {html_path}")
            print(f"HTML file generated at: {html_path}")
            # response = send_file(
            #     html_path,
            #     as_attachment=True,
            #     download_name=html_file,
            #     mimetype="text/html"
            # )
            # return response

        else:
            # Return JSON if requested
            print("Returning materials as JSON.")
            # return jsonify({"success": True, "materials": materials})

    except Exception as e:
        print(f"Error occurred in generate_materials: {str(e)}")
        # return jsonify({"error": str(e)}), 500
