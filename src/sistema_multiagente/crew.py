from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import json
import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

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

class ArticleSection(BaseModel):
    """Modelo para seção do artigo"""
    title: str
    content: str

class WrittenArticle(BaseModel):
    """Modelo para o artigo escrito"""
    title: str
    introduction: str
    sections: List[ArticleSection] = []
    conclusion: str
    word_count: int

class EditedArticle(BaseModel):
    """Modelo para o artigo editado"""
    title: str
    introduction: str
    sections: List[ArticleSection] = []
    conclusion: str
    word_count: int
    improvements: List[str]
    
    def to_markdown(self) -> str:
        """
        Converte o artigo para formato markdown
        
        Returns:
            str: Artigo formatado em markdown
        """
        markdown = f"# {self.title}\n\n"
        markdown += f"{self.introduction}\n\n"
        
        for section in self.sections:
            markdown += f"## {section.title}\n\n"
            markdown += f"{section.content}\n\n"
            
        markdown += f"## Conclusão\n\n{self.conclusion}\n\n"
        
        markdown += f"*Este artigo contém {self.word_count} palavras.*\n\n"
        
        return markdown

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
        response.raise_for_status()  # Lança exceção para respostas com erro
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
        response.raise_for_status()  # Lança exceção para respostas com erro
        data = response.json()
        
        if "query" in data and "pages" in data["query"]:
            page = data["query"]["pages"].get(page_id)
            if page and "extract" in page:
                return page["extract"]
            return "Não foi possível obter o conteúdo."
        return "Página não encontrada."
    except Exception as e:
        return f"Erro ao obter conteúdo: {str(e)}"

@tool("Contar palavras")
def count_words(text: str) -> int:
    """
    Conta o número de palavras em um texto.
    
    Args:
        text: Texto para contar palavras
        
    Returns:
        Número de palavras
    """
    if not text:
        return 0
    words = text.split()
    return len(words)

@CrewBase
class SistemaMultiagente():
    """
    SistemaMultiagente crew para geração de artigos com base em pesquisas na Wikipedia.
    
    Esta classe implementa um sistema multiagente usando CrewAI para:
    1. Pesquisar informações sobre um tópico na Wikipedia
    2. Escrever um artigo com base nas informações coletadas
    3. Revisar e editar o artigo para melhorar sua qualidade
    
    Attributes:
        agents_config (str): Caminho para o arquivo de configuração dos agentes
        tasks_config (str): Caminho para o arquivo de configuração das tarefas
        llm_provider (str): Provedor de LLM a ser utilizado ('openai', 'groq', 'gemini')
        api_key (str): Chave de API para o provedor de LLM
    """

    # Configuração via YAML
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self, llm_provider: str = 'local'):
        """
        Inicializa o sistema multiagente.
        
        Args:
            llm_provider: Provedor de LLM a ser utilizado ('openai', 'groq', 'gemini', 'local')
        """
        self.llm_provider = llm_provider
        
        # Configurar LLM com base no provedor
        if llm_provider == 'groq':
            self.api_key = os.getenv('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not set in environment variables")
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama3-70b-8192"
            )
        elif llm_provider == 'gemini':
            self.api_key = os.getenv('GOOGLE_API_KEY')
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment variables")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model="gemini-1.5-pro"
            )
        elif llm_provider == 'local':
            # Use a local model provider like Ollama or a similar option
            print("Using local LLM provider - no API key required")
            from langchain_ollama import OllamaLLM
            try:
                self.llm = OllamaLLM(
                    model="ollama/llama3.1",  # Add the "ollama/" prefix to the model name
                    base_url="http://localhost:11434",
                    temperature=0.7,
                    timeout=180  # Changed from request_timeout to timeout
                )
            except Exception as e:
                print(f"Error connecting to Ollama: {str(e)}")
                print("Make sure the Ollama server is running (ollama serve)")
                print("Make sure you've pulled the llama3.1 model with: ollama pull llama3.1")
                raise ValueError(f"Failed to initialize Ollama: {str(e)}")
        else:
            # Usar OpenAI como padrão
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            self.llm = None  # CrewAI usa OpenAI por padrão

    @agent
    def researcher(self) -> Agent:
        """
        Cria um agente pesquisador que coleta informações sobre o tópico.
        
        Returns:
            Agent: O agente pesquisador configurado
        """
        agent_config = {
            "role": "Pesquisador",
            "goal": "Pesquisar e coletar informações relevantes sobre o tema {topic}",
            "backstory": "Sou especialista em pesquisa e análise de informações. Minha missão "
                         "é encontrar conteúdo relevante e confiável para a produção de artigos."
        }
        
        return Agent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            verbose=True,
            tools=[search_wikipedia, get_wikipedia_content],
            llm=self.llm,
            allow_delegation=True
        )

    @agent
    def writer(self) -> Agent:
        """
        Cria um agente escritor que transforma as pesquisas em um artigo coeso.
        
        Returns:
            Agent: O agente escritor configurado
        """
        agent_config = {
            "role": "Redator",
            "goal": "Criar artigos informativos e bem estruturados com base nas informações coletadas",
            "backstory": "Sou um redator experiente especializado em transformar informações brutas "
                         "em conteúdo atraente e informativo para websites."
        }
        
        return Agent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            verbose=True,
            tools=[count_words],
            llm=self.llm,
            allow_delegation=True
        )
    
    @agent
    def editor(self) -> Agent:
        """
        Cria um agente editor que revisa e melhora o artigo final.
        
        Returns:
            Agent: O agente editor configurado
        """
        agent_config = {
            "role": "Editor",
            "goal": "Revisar e aprimorar o artigo final, verificando clareza, coesão e gramática.",
            "backstory": "Sou um editor meticuloso com olhar crítico para garantir a qualidade final "
                         "dos artigos publicados."
        }
        
        return Agent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            verbose=True,
            tools=[count_words],
            llm=self.llm,
            allow_delegation=True
        )

    @task
    def research_task(self) -> Task:
        """
        Define a tarefa de pesquisa.
        
        Returns:
            Task: A tarefa de pesquisa configurada
        """
        task_config = {
            "description": "Pesquise informações completas sobre '{topic}'. Identifique de 3 a 5 artigos "
                          "relevantes da Wikipedia, escolha os mais apropriados e extraia seu conteúdo. "
                          "Organize as informações em tópicos principais.",
            "expected_output": "Um resumo estruturado das informações coletadas, com os principais tópicos e "
                               "fatos sobre o tema. Retorne um objeto ResearchResult com os artigos e resumo."
        }
        
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.researcher()
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
        task_config = {
            "description": "Com base nas informações fornecidas pelo Pesquisador, escreva um artigo informativo "
                          "com no mínimo 300 palavras. O artigo deve ter introdução, desenvolvimento e conclusão. "
                          "Inclua subtítulos relevantes e organize o conteúdo de forma lógica. "
                          "Retorne um objeto WrittenArticle com o artigo estruturado.",
            "expected_output": "Um artigo bem estruturado com no mínimo 300 palavras sobre o tema solicitado."
        }
        
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.writer(),
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
        task_config = {
            "description": "Revise o artigo fornecido pelo Redator. Verifique a clareza, coesão, gramática e formatação. "
                          "Faça as correções necessárias e sugira melhorias quando apropriado. "
                          "Retorne um objeto EditedArticle com o artigo revisado e as melhorias realizadas.",
            "expected_output": "O artigo final revisado e pronto para publicação, no formato EditedArticle."
        }
        
        return Task(
                description=task_config["description"],
                expected_output=task_config["expected_output"],
                agent=self.editor(),
                context=context,
                output_file="artigo.md"
        )
    
    @crew
    def create_crew(self) -> Crew:
        """
        Cria a tripulação de agentes e configura seu fluxo de trabalho.
        
        Returns:
            Crew: A tripulação configurada
        """
        # Create agents
        researcher_agent = self.researcher()
        writer_agent = self.writer()
        editor_agent = self.editor()
        
        # Create tasks without explicitly setting context
        research = self.research_task()
        writing = self.writing_task()
        editing = self.editing_task()
        
        # Define the workflow - the sequential process will handle the dependencies
        return Crew(
            agents=[researcher_agent, writer_agent, editor_agent],
            tasks=[research, writing, editing],
            process=Process.sequential,
            verbose=True,
        )
    
    def run(self, topic: str) -> EditedArticle:
        """
        Executa o fluxo completo do sistema multiagente.
        
        Args:
            topic: Tópico para pesquisa e escrita do artigo
            
        Returns:
            EditedArticle: Artigo final editado
        """
        inputs = {
            'topic': topic
        }
        
        # Add better error handling and logging
        try:
            result = self.create_crew().kickoff(inputs=inputs)
            
            # Log the result type and content for debugging
            print(f"Crew result type: {type(result)}")
            print(f"Crew result content: {result}")
            
            # Handle different result types
            if result is None:
                # Create a default article when result is None
                return EditedArticle(
                    title=f"Artigo sobre {topic}",
                    introduction="Não foi possível gerar o artigo.",
                    sections=[],
                    conclusion="Erro no processamento do fluxo de trabalho.",
                    word_count=0,
                    improvements=["Falha na geração do artigo"]
                )
            elif isinstance(result, EditedArticle):
                return result
            elif isinstance(result, dict):
                return EditedArticle(**result)
            elif isinstance(result, str):
                # Try to process string as JSON first
                try:
                    data = json.loads(result)
                    return EditedArticle(**data)
                except json.JSONDecodeError:
                    # If not JSON, create a basic article
                    return EditedArticle(
                        title=f"Artigo sobre {topic}",
                        introduction="Introdução gerada automaticamente",
                        sections=[],
                        conclusion=result,
                        word_count=len(result.split()),
                        improvements=["Artigo gerado com formato não estruturado"]
                    )
            else:
                # For any other type
                return EditedArticle(
                    title=f"Artigo sobre {topic}",
                    introduction="Resultado em formato não esperado",
                    sections=[],
                    conclusion=str(result),
                    word_count=len(str(result).split()),
                    improvements=["Formato de resultado não suportado"]
                )
        except Exception as e:
            # Log the exception
            print(f"Error in crew execution: {e}")
            # Return a fallback article
            return EditedArticle(
                title=f"Erro na geração do artigo sobre {topic}",
                introduction="Ocorreu um erro durante a geração do artigo.",
                sections=[ArticleSection(
                    title="Detalhes do erro",
                    content=str(e)
                )],
                conclusion="Por favor, tente novamente mais tarde.",
                word_count=0,
                improvements=["Resolver o erro: " + str(e)]
            )