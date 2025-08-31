from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class Topic(BaseModel):
    """Model for extracted topics from syllabus."""
    title: str = Field(description="The title of the topic")
    description: str = Field(description="Brief description of the topic")
    subtopics: Optional[List[str]] = Field(default_factory=list, description="List of subtopics if any")

class Topics(BaseModel):
    """Container for multiple topics."""
    topics: List[Topic] = Field(description="List of topics from the syllabus")

def get_topic_extraction_prompt():
    """Prompt for extracting topics from syllabus content."""
    template = """Extract the main topics from this syllabus content. For each topic, provide a title with the Module no,
    brief description, and any subtopics mentioned with unit no.

    Syllabus: {syllabus_content}

    Format the output as a JSON list of topics with their descriptions and subtopics.
    {format_instructions}
    """

    return PromptTemplate(
        template=template,
        input_variables=["syllabus_content"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Topics).get_format_instructions()}
    )
