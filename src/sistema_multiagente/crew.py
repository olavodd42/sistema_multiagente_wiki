from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import requests
import json

class WikipediaArticle(BaseModel):
    """Modelo para artigos da Wikipedia"""
    title: str
    content: str
    pageid: str

class ResearchResult(BaseModel):
    """Modelo para resultados de pesquisa"""
    topic: str
    articles: List[WikipediaArticle]
    summary: str

class WrittenArticle(BaseModel):
    """Modelo para o artigo escrito"""
    title: str
    introduction: str
    body: str
    conclusion: str
    word_count: int

class EditedArticle(BaseModel):
    """Modelo para o artigo editado"""
    title: str
    introduction: str
    body: str
    conclusion: str
    word_count: int
    improvements: List[str]

# Definindo as ferramentas como funções decoradas com @tool
@tool("Buscar na Wikipedia")
def search_wikipedia(query: str) -> str:
    """
    Busca por artigos na Wikipedia relacionados ao termo de pesquisa.
    
    Args:
        query: Termo de busca
        
    Returns:
        Resultado da busca em formato JSON string
    """
    try:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "utf8": 1
        }
        response = requests.get("https://pt.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        if "query" in data and "search" in data["query"]:
            results = data["query"]["search"]
            results_list = [{"title": item["title"], "pageid": str(item["pageid"])} for item in results]
            return json.dumps(results_list)
        return "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro na busca: {str(e)}"

@tool("Obter conteúdo da Wikipedia")
def get_wikipedia_content(page_id: str) -> str:
    """
    Obtém o conteúdo de uma página da Wikipedia pelo seu ID.
    
    Args:
        page_id: ID da página da Wikipedia
        
    Returns:
        Conteúdo da página
    """
    try:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True,
            "exsectionformat": "plain",
            "utf8": 1
        }
        response = requests.get("https://pt.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        if "query" in data and "pages" in data["query"]:
            page = data["query"]["pages"].get(page_id)
            if page and "extract" in page:
                return page["extract"]
            return "Não foi possível obter o conteúdo."
        return "Página não encontrada."
    except Exception as e:
        return f"Erro ao obter conteúdo: {str(e)}"

@CrewBase
class SistemaMultiagente():
    """
    SistemaMultiagente crew para geração de artigos com base em pesquisas na Wikipedia.
    
    Esta classe implementa um sistema multiagente usando CrewAI para:
    1. Pesquisar informações sobre um tópico na Wikipedia
    2. Escrever um artigo com base nas informações coletadas
    3. Revisar e editar o artigo para melhorar sua qualidade
    """

    # Configuração via YAML
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        """
        Cria um agente pesquisador que coleta informações sobre o tópico.
        
        Returns:
            Agent: O agente pesquisador configurado
        """
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[search_wikipedia, get_wikipedia_content],
            allow_delegation=True
        )

    @agent
    def writer(self) -> Agent:
        """
        Cria um agente escritor que transforma as pesquisas em um artigo coeso.
        
        Returns:
            Agent: O agente escritor configurado
        """
        return Agent(
            config=self.agents_config['writer'],
            verbose=True,
            allow_delegation=True
        )
    
    @agent
    def editor(self) -> Agent:
        """
        Cria um agente editor que revisa e melhora o artigo final.
        
        Returns:
            Agent: O agente editor configurado
        """
        return Agent(
            config=self.agents_config['editor'],
            verbose=True,
            allow_delegation=True
        )

    @task
    def research_task(self) -> Task:
        """
        Define a tarefa de pesquisa.
        
        Returns:
            Task: A tarefa de pesquisa configurada
        """
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def writing_task(self, context=None) -> Task:
        """
        Define a tarefa de escrita.
        
        Args:
            context: Contexto opcional da tarefa anterior
            
        Returns:
            Task: A tarefa de escrita configurada
        """
        return Task(
            config=self.tasks_config['writing_task'],
            context=context
        )
    
    @task
    def editing_task(self, context=None) -> Task:
        """
        Define a tarefa de edição.
        
        Args:
            context: Contexto opcional da tarefa anterior
            
        Returns:
            Task: A tarefa de edição configurada
        """
        return Task(
            config=self.tasks_config['editing_task'],
            context=context
        )

    @crew
    def crew(self) -> Crew:
        """
        Cria a tripulação de agentes e configura seu fluxo de trabalho.
        
        Returns:
            Crew: A tripulação configurada
        """
        # Create tasks
        research = Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )
        
        writing = Task(
            config=self.tasks_config['writing_task'],
            agent=self.writer(),
            dependencies=[research]
        )
        
        editing = Task(
            config=self.tasks_config['editing_task'],
            agent=self.editor(),
            dependencies=[writing]
        )
        
        return Crew(
            agents=[self.researcher(), self.writer(), self.editor()],
            tasks=[research, writing, editing],
            process=Process.sequential,
            verbose=True,
        )