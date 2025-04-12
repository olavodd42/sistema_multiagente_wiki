#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uvicorn
import json
from sistema_multiagente.crew import SistemaMultiagente, EditedArticle, ArticleSection

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Define Pydantic models para API
class ArticleRequest(BaseModel):
    """Model for article generation request"""
    topic: str = Field(..., description="Topic to research and write about")
    min_words: int = Field(300, description="Minimum word count for the article")
    llm_provider: str = Field("groq", description="LLM provider (groq, gemini, openai)")
    
class ArticleResponse(BaseModel):
    """Model for article generation response"""
    task_id: str
    status: str
    message: str

class ArticleTopic(BaseModel):
    """Model for a Wikipedia article topic"""
    title: str
    pageid: str

class TaskStatus(BaseModel):
    """Model for task status"""
    task_id: str
    status: str
    message: str
    
class ArticleResult(BaseModel):
    """Model for the complete article result"""
    title: str
    introduction: str
    sections: List[ArticleSection]
    conclusion: str
    word_count: int
    improvements: Optional[List[str]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Sistema Multiagente API", 
    description="API para geração de artigos utilizando um sistema multiagente com CrewAI",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store results for background tasks
results: Dict[str, Any] = {}

def run_task(topic: str, task_id: str, llm_provider: str = "groq"):
    """
    Run the crew with the given topic and store results.
    
    Args:
        topic: The topic to research and write about
        task_id: Unique identifier for this task
        llm_provider: LLM provider to use (groq, gemini, openai)
    """
    try:
        # Atualizar status para "processing"
        results[task_id] = {
            "status": "processing", 
            "message": "Gerando artigo. Isso pode levar alguns minutos..."
        }
        
        # Criar sistema multiagente com o provedor especificado
        sistema = SistemaMultiagente(llm_provider=llm_provider)
        
        # Executar a crew e obter resultado
        resultado = sistema.run(topic)
        
        # Salvar resultado em arquivo
        markdown_file = f"artigos/{task_id}.md"
        os.makedirs("artigos", exist_ok=True)
        
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(resultado.to_markdown())
        
        # Armazenar resultado
        results[task_id] = {
            "status": "completed",
            "article": {
                "title": resultado.title,
                "introduction": resultado.introduction,
                "sections": [{"title": s.title, "content": s.content} for s in resultado.sections],
                "conclusion": resultado.conclusion,
                "word_count": resultado.word_count,
                "improvements": resultado.improvements
            },
            "file_path": markdown_file
        }
    except Exception as e:
        # Em caso de erro
        results[task_id] = {"status": "error", "message": str(e)}

@app.post("/generate-article", response_model=ArticleResponse)
async def generate_article(article_request: ArticleRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to generate an article asynchronously
    
    Args:
        article_request: Request containing the topic and minimum word count
        background_tasks: FastAPI background tasks
    
    Returns:
        Dictionary with task ID and status information
    """
    task_id = f"task_{datetime.now().timestamp()}"
    
    # Inicializar resultado
    results[task_id] = {
        "status": "queued", 
        "message": "Article generation queued. Check status with /article/{task_id}"
    }
    
    # Adicionar tarefa em background
    background_tasks.add_task(
        run_task, 
        article_request.topic, 
        task_id,
        article_request.llm_provider
    )
    
    return {
        "task_id": task_id, 
        "status": "queued", 
        "message": "Article generation queued. Check status with /article/{task_id}"
    }

@app.get("/article/{task_id}", response_model=Dict[str, Any])
async def get_article(task_id: str):
    """
    Endpoint to retrieve a generated article by task ID
    
    Args:
        task_id: Unique identifier for the task
    
    Returns:
        Article data or status message
    """
    if task_id not in results:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    return results[task_id]

@app.get("/article/{task_id}/download")
async def download_article(task_id: str):
    """
    Endpoint to download the generated article as markdown
    
    Args:
        task_id: Unique identifier for the task
    
    Returns:
        Markdown file
    """
    if task_id not in results:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    result = results[task_id]
    
    if result.get("status") != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Article generation is not completed. Current status: {result.get('status')}"
        )
        
    file_path = result.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Article file not found")
        
    return FileResponse(
        path=file_path,
        filename=f"artigo_{task_id}.md",
        media_type="text/markdown"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}

def cli_run(topic=None, llm_provider="groq"):
    """
    Run the crew via CLI.
    
    Args:
        topic: The topic to research (optional)
        llm_provider: LLM provider to use (groq, gemini, openai)
    """
    if topic is None:
        topic = "Inteligência Artificial e seus impactos na sociedade"
    
    try:
        sistema = SistemaMultiagente(llm_provider=llm_provider)
        result = sistema.run(topic)
        
        # Salvar resultado em arquivo
        os.makedirs("artigos", exist_ok=True)
        file_path = f"artigos/artigo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result.to_markdown())
            
        print(f"\n\nArticle generated successfully!")
        print(f"Output written to: {os.path.abspath(file_path)}")
        
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def serve(host="0.0.0.0", port=8000):
    """
    Start the FastAPI server
    
    Args:
        host: Host to bind the server
        port: Port to bind the server
    """
    uvicorn.run("sistema_multiagente.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    serve()