"""
Content processing and material generation for the RAG system.
"""

import os
import re
from typing import Optional, List, Dict, Any
from logger import logger

from ..core import BaseRAGComponent, RAGConfig

class ContentProcessor(BaseRAGComponent):
    """Handles content processing and validation."""

    def fix_mermaid_syntax(self, content: str) -> str:
        """
        Fix common Mermaid syntax errors in the generated content.
        """
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

    def validate_mermaid_syntax(self, content: str) -> tuple[bool, List[str]]:
        """
        Validate Mermaid syntax and return True if valid, False if errors found.
        """
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

class MaterialGenerator(BaseRAGComponent):
    """Handles material generation and file output."""

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.content_processor = ContentProcessor(config)

    def generate_materials(self, topic: str, teacher_id: str, output_format: str = 'pdf') -> Optional[str]:
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
            self._log_operation("Starting material generation", f"topic='{topic}', format='{output_format}'")

            # This would integrate with the main RAG system
            # For now, return a placeholder
            return f"Generated {output_format} for topic: {topic}"

        except Exception as e:
            self._log_error("Material generation", e)
            return None

    def _create_markdown_content(self, topic: str, content: str) -> str:
        """Create formatted markdown content for Marp."""
        from config import config

        # Use configuration for Marp settings
        marp_config = config.MARP_CONFIG
        return f'''---
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
\n'''

class FileManager(BaseRAGComponent):
    """Handles file operations and output generation."""

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        from config import config as app_config
        self.upload_folder = app_config.UPLOAD_FOLDER

    def ensure_upload_directory(self):
        """Ensure upload directory exists."""
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
            self._log_operation("Created upload directory", self.upload_folder)

    def generate_output_path(self, topic: str, output_format: str) -> str:
        """Generate output file path."""
        topic_clean = topic.replace(' ', '_').lower()
        base_name = f"{topic_clean}_materials"

        if output_format == 'pdf':
            filename = f"{base_name}.pdf"
        elif output_format in ['pptx', 'ppt']:
            filename = f"{base_name}_slides.pptx"
        elif output_format == 'html':
            filename = f"{base_name}.html"
        else:
            filename = f"{base_name}.md"

        return os.path.join(self.upload_folder, filename)
