from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class Topic(BaseModel):
    """Model for extracted topics from syllabus."""
    module_number: Optional[str] = Field(default=None, description="Module number (e.g., '1', '2')")
    unit_number: Optional[str] = Field(default=None, description="Unit number (e.g., '1.1', '1.2')")
    title: str = Field(description="The title of the topic")
    description: str = Field(description="Brief description of the topic")
    subtopics: Optional[List[str]] = Field(default_factory=list, description="List of subtopics if any")

class Topics(BaseModel):
    """Container for multiple topics."""
    topics: List[Topic] = Field(description="List of topics from the syllabus")

def get_topic_extraction_prompt():
    """Prompt for extracting topics from syllabus content with module/unit structure."""
    template = """Extract the main topics from this syllabus content. For each topic, identify:

1. Module number (e.g., "1", "2", "3") - extract from lines starting with numbers like "1 Title"
2. Unit number (e.g., "1.1", "1.2", "2.1") - extract from lines starting with numbers like "1.1", "1.2"
3. Topic title - the main heading or title of the topic
4. Brief description - summarize what the topic covers
5. Any subtopics mentioned - list specific subtopics under each unit

Syllabus: {syllabus_content}

IMPORTANT INSTRUCTIONS:
- Look for patterns like "1 Title", "1.1 Definition", "1.2 Hardware Concepts"
- Module numbers are the first number (e.g., "1" in "1 Title" or "1.1 Definition")
- Unit numbers include the decimal (e.g., "1.1", "1.2")
- If a line starts with "1 Title", that's module 1
- If a line starts with "1.1 Definition", that's unit 1.1 under module 1
- Group related content under the appropriate module/unit
- Extract all modules and units mentioned in the syllabus

Format the output as a JSON list of topics with their module/unit information, descriptions and subtopics.
{format_instructions}
"""

    return PromptTemplate(
        template=template,
        input_variables=["syllabus_content"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Topics).get_format_instructions()}
    )
