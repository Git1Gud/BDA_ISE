from langchain.prompts import PromptTemplate

def get_study_material_prompt():
    """Prompt for generating comprehensive study materials."""
    template = """Create comprehensive, information-dense slides in Marp-compatible markdown for:

Topic: {topic} (This is your topic you want to generate material if it is a number like module 1 or unit 1 then fallback to description otherwise dont use it)
Description: {description}

Based on these detailed reference materials:
{reference_content}

Content Requirements:
- Create approx 20 slides
- Include extensive factual content, definitions, and explanations
- Cover the topic thoroughly with academic depth
- Include relevant theories, methodologies, historical context, and current applications
- Define all technical terminology completely
- Incorporate statistics, research findings, and scholarly perspectives
- Include key examples that demonstrate practical applications
- Provide comprehensive explanations of complex concepts


Formatting Guidelines:
- Begin each slide with '---'
- Please add ```mermaid block when using mermaid diagrams and end with ```. I will be using it in my code to convert mermaid diagrams. And please do not add ``` anywhere except mermaid. A humble request

- PLEASE DO NOT ADD ```markdown
- Make sure the slide do no overflow
- Use hierarchical headings to organize dense information (## for main titles, ### for subtitles)
- Employ multi-level bullet points for detailed breakdowns
- Format each slide to maximize information while maintaining readability
- Use **bold** and *italics* to highlight critical terms and concepts
- ADD reference at the end . Make sure the references are valid and do no use your own brain (which you ofcourse do not have)
- ADD Atleast 1 MERMAID DIAGRAMS. Also make sure the diagram is small to fit and does not overflow. Also start your diagram with ```mermaid and end with ```.

DO NOT:
- DO NOT ADD ```markdown
- Add Marp metadata (I'll handle that separately)
- Include image references
- Write annotations like "Here is the slide content"
- Sacrifice depth for brevity
- Omit important details or nuances
- DO NOT ADD ``` markdown code block
- DO NOT ADD ``` code block
    """

    return PromptTemplate(
        template=template,
        input_variables=["topic", "description", "reference_content", "teacher_id"]
    )
