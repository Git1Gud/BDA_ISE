# from flask import request, jsonify, send_file
import os
from RAG import StudyMaterialRAG

study_system=StudyMaterialRAG()

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
        theme: default
        paginate: true
        \n''')
            for subtopic, content in materials.items():
                f.write('---\n')
                f.write(f"### {subtopic}\n\n")
                
                # Check if the content contains mermaid diagrams
                if '```mermaid' in content:
                    # Process the content to preserve mermaid blocks but remove other code blocks
                    lines = content.split('\n')
                    in_mermaid_block = False
                    clean_lines = []
                    
                    for line in lines:
                        if '```mermaid' in line:
                            in_mermaid_block = True
                            clean_lines.append(line)
                        elif '```' in line and in_mermaid_block:
                            in_mermaid_block = False
                            clean_lines.append(line)
                        elif in_mermaid_block:
                            clean_lines.append(line)
                        else:
                            # Clean non-mermaid lines
                            clean_line = line.replace("```markdown", "").replace("```", "")
                            clean_lines.append(clean_line)
                            
                    clean_content = '\n'.join(clean_lines)
                else:
                    # If no mermaid blocks, remove all code block markers
                    clean_content = content.replace("```markdown", "").replace("```", "").strip()
                    
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
            os.system(f"marp --allow-local-files {file_path} --pdf -o {html_path}")
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
