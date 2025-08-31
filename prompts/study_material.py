from langchain.prompts import PromptTemplate

def get_study_material_prompt():
    """Prompt for generating comprehensive study materials."""
    template = """Create comprehensive, information-dense slides in Marp-compatible markdown for:

Topic: {topic} (This is your topic you want to generate material if it is a number like module 1 or unit 1 then fallback to description otherwise dont use it)
Description: {description}

Based on these detailed reference materials:
{reference_content}

STYLING CONTEXT (IMPORTANT - Read this carefully):
{css_context}

Content Requirements:
- Create approx 20 slides
- Include extensive factual content, definitions, and explanations
- Cover the topic thoroughly with academic depth
- Include relevant theories, methodologies, historical context, and current applications
- Define all technical terminology completely
- Incorporate statistics, research findings, and scholarly perspectives
- Include key examples that demonstrate practical applications
- Provide comprehensive explanations of complex concepts
- CRITICAL: Include at least 2-3 relevant Mermaid diagrams throughout the presentation


Formatting Guidelines:
- Begin each slide with '---'
- Please add ```mermaid block when using mermaid diagrams and end with ```. I will be using it in my code to convert mermaid diagrams. And please do not add ``` anywhere except mermaid. A humble request

- PLEASE DO NOT ADD ```markdown
- CRITICAL: Ensure ALL content fits within slide boundaries - no text or diagrams should overflow
- Keep Mermaid diagrams compact and readable - use maximum width of 800px and height of 400px
- Limit text content to 8-12 bullet points per slide maximum
- Use concise language and break long sentences into shorter ones
- Keep headings short and descriptive (max 50 characters)
- Use hierarchical headings to organize dense information (## for main titles, ### for subtitles)
- Employ multi-level bullet points for detailed breakdowns
- Format each slide to maximize information while maintaining readability
- Use **bold** and *italics* to highlight critical terms and concepts
- ADD reference at the end . Make sure the references are valid and do no use your own brain (which you ofcourse do not have)
- ADD Atleast 2-3 MERMAID DIAGRAMS. Also make sure the diagram is small to fit and does not overflow. Also start your diagram with ```mermaid and end with ```.

CRITICAL MERMAID SYNTAX RULES:
- NEVER use curly braces in Mermaid diagrams - they will be interpreted as template variables
- For decision nodes, use DOUBLE SQUARE BRACKETS [[Decision Point]] instead of curly braces Decision Point 
- For process nodes, use square brackets [Process Name] instead of parentheses (Process Name)
- Always use proper Mermaid syntax: --> for arrows, [Node] for rectangles, ((Circle)) for circles
- Example: A[Start] --> B[[Decision Point]] --> C[Action 1]
- AVOID: A(Start) --> B(curly braces)Decision Point(curly braces) --> C(Action 1)

Mermaid Diagram Guidelines (CRITICAL):
- Include at least 2-3 different types of Mermaid diagrams (flowcharts, sequence diagrams, Gantt charts, etc.)
- Place diagrams strategically to illustrate key concepts
- Keep diagrams simple and readable - avoid overly complex diagrams
- Use clear, descriptive labels for all diagram elements
- Ensure diagrams complement the text content
- Maximum 1 diagram per slide to avoid overcrowding
- Use appropriate diagram types for the content (flowchart for processes, sequence for interactions, etc.)

CORRECT MERMAID SYNTAX EXAMPLES (USE THESE EXACT FORMATS):

Flowchart Example:
```mermaid
graph TD
    A[Start] --> B[[Decision_Point]]
    B -->|Yes| C[Action_1]
    B -->|No| D[Action_2]
    C --> E[End]
    D --> E
```

Sequence Diagram Example:
```mermaid
sequenceDiagram
    participant Client
    participant Server
    Client->>Server: Request
    Server-->>Client: Response
```

Gantt Chart Example:
```mermaid
gantt
    title Project Timeline
    dateFormat YYYY-MM-DD
    section Planning
    Task_1          :done,    t1, 2024-01-01, 2024-01-05
    section Development
    Task_2          :active,  t2, 2024-01-06, 2024-01-15
```

IMPORTANT: Always use correct Mermaid syntax:
- Use 'graph TD' for top-down flowcharts, 'graph LR' for left-right
- Use proper arrow syntax: --> for solid arrows, -.-> for dotted arrows
- Use square brackets [ ] for nodes, not parentheses ( )
- For sequence diagrams, use '->>' for solid arrows, '-->>' for dotted arrows
- Do NOT use 'PS', 'PE', or other invalid syntax elements
- Always test diagram syntax mentally before including

AVOID THESE COMMON ERRORS:
- Don't use ( ) for node labels - use [ ] instead
- Don't use 'PS' or 'PE' syntax - not valid Mermaid
- Don't use invalid arrow types
- Don't forget to close diagram blocks with ```

Content Structure Guidelines:
- Each slide should have 1 main heading and 3-8 sub-points
- Avoid walls of text - use bullet points extensively
- Keep individual bullet points under 100 characters
- Use abbreviations and acronyms where appropriate to save space
- Break complex concepts into multiple slides if needed
- Balance text content with visual elements (diagrams)

DO NOT:
- DO NOT ADD ```markdown
- Add Marp metadata (I'll handle that separately)
- Include image references or any image tags
- Write annotations like "Here is the slide content"
- Create slides with more than 15 lines of content
- Use extremely long words that might cause wrapping issues
- Sacrifice depth for brevity
- Omit important details or nuances
- DO NOT ADD ``` markdown code block
- DO NOT ADD ``` code block
- DO NOT include any HTML image tags or image references
    """

    return PromptTemplate(
        template=template,
        input_variables=["topic", "description", "reference_content", "css_context", "teacher_id"]
    )
