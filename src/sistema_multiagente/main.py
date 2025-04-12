#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
from sistema_multiagente.crew import SistemaMultiagente

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Define Pydantic models for API input and output
class ArticleRequest(BaseModel):
    """Model for article generation request"""
    topic: str = Field(..., description="Topic to research and write about")
    min_words: Optional[int] = Field(300, description="Minimum word count for the article")
    
class ArticleTopic(BaseModel):
    """Model for a Wikipedia article topic"""
    title: str
    pageid: str

class ArticleSection(BaseModel):
    """Model for an article section"""
    title: str
    content: str

class Article(BaseModel):
    """Model for the generated article"""
    title: str
    introduction: str
    sections: List[ArticleSection]
    conclusion: str
    word_count: int

# Initialize FastAPI app
app = FastAPI(title="Sistema Multiagente API", 
              description="API para geração de artigos utilizando um sistema multiagente com CrewAI")

# Store results for background tasks
results = {}

def run(topic: str, task_id: str):
    """
    Run the crew with the given topic and store results.
    
    Args:
        topic: The topic to research and write about
        task_id: Unique identifier for this task
    """
    inputs = {
        'topic': topic,
        'current_year': str(datetime.now().year)
    }
    
    try:
        resultado = SistemaMultiagente().crew().kickoff(inputs=inputs)
        results[task_id] = resultado
    except Exception as e:
        results[task_id] = {"error": str(e)}

@app.post("/generate-article", response_model=dict)
async def generate_article(article_request: ArticleRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to generate an article asynchronously
    
    Args:
        article_request: Request containing the topic and minimum word count
        background_tasks: FastAPI background tasks
    
    Returns:
        Dictionary with task ID
    """
    task_id = f"task_{datetime.now().timestamp()}"
    background_tasks.add_task(run, article_request.topic, task_id)
    return {"task_id": task_id, "status": "processing", "message": "Article generation started"}

@app.get("/article/{task_id}", response_model=dict)
async def get_article(task_id: str):
    """
    Endpoint to retrieve a generated article by task ID
    
    Args:
        task_id: Unique identifier for the task
    
    Returns:
        Article data or status message
    """
    if task_id in results:
        return results[task_id]
    return {"status": "not_found", "message": "Task ID not found or still processing"}


def cli_run(topic=None):
    """
    Run the crew via CLI.
    
    Args:
        topic: The topic to research (optional)
    """
    if topic is None:
        topic = "Inteligência Artificial e seus impactos na sociedade"
        
    inputs = {
        'topic': topic,
        'current_year': str(datetime.now().year)
    }
    
    try:
        result = SistemaMultiagente().crew().kickoff(inputs=inputs)
        print(f"\n\nArticle generated successfully!")
        print(f"Output written to: {os.path.join(os.getcwd(), 'artigo.md')}")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    
def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        SistemaMultiagente().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        SistemaMultiagente().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    serve()
