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

"""
Correções para a classe EditedArticle em src/sistema_multiagente/crew.py
"""

class EditedArticle(BaseModel):
    """Modelo para o artigo editado"""
    title: str
    introduction: str
    sections: List[ArticleSection] = Field(default_factory=list)
    conclusion: str
    word_count: int
    improvements: List[str] = Field(default_factory=list)
    
    def to_markdown(self) -> str:
        """
        Converte o artigo para formato markdown
        
        Returns:
            str: Artigo formatado em markdown
        """
        try:
            # Verifica se os campos estão presentes e têm valor
            title = self.title if self.title else "Artigo Sem Título"
            introduction = self.introduction if self.introduction else "Sem introdução."
            conclusion = self.conclusion if self.conclusion else "Sem conclusão."
            
            # Inicia o markdown com o título e introdução
            markdown = f"# {title}\n\n"
            markdown += f"{introduction}\n\n"
            
            # Adiciona seções se existirem
            if self.sections and len(self.sections) > 0:
                for section in self.sections:
                    section_title = section.title if section.title else "Seção"
                    section_content = section.content if section.content else "Conteúdo não disponível."
                    markdown += f"## {section_title}\n\n"
                    markdown += f"{section_content}\n\n"
            else:
                # Se não há seções, adiciona uma mensagem
                markdown += "## Conteúdo\n\n"
                markdown += "Não há seções disponíveis para este artigo.\n\n"
            
            # Adiciona conclusão
            markdown += f"## Conclusão\n\n{conclusion}\n\n"
            
            # Adiciona contagem de palavras
            word_count = self.word_count if self.word_count > 0 else len(markdown.split())
            markdown += f"*Este artigo contém {word_count} palavras.*\n\n"
            
            # Adiciona lista de melhorias, se disponível
            if self.improvements and len(self.improvements) > 0:
                markdown += "## Melhorias Realizadas\n\n"
                for improvement in self.improvements:
                    markdown += f"- {improvement}\n"
                markdown += "\n"
            
            return markdown
        except Exception as e:
            # Em caso de erro, retorna uma mensagem de erro com o artigo básico
            error_markdown = f"# Erro ao formatar o artigo\n\n"
            error_markdown += f"Ocorreu um erro ao converter o artigo para markdown: {str(e)}\n\n"
            error_markdown += f"## Conteúdo original\n\n"
            error_markdown += f"Título: {self.title}\n\n"
            error_markdown += f"Introdução: {self.introduction}\n\n"
            error_markdown += f"Conclusão: {self.conclusion}\n\n"
            
            return error_markdown
        

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
            "description": """
            Pesquise informações completas sobre '{topic}'. Identifique de 3 a 5 artigos 
            relevantes da Wikipedia, escolha os mais apropriados e extraia seu conteúdo. 
            Organize as informações em tópicos principais.
            
            IMPORTANTE: Seu resultado deve ser um objeto ResearchResult estruturado com os seguintes campos:
            - topic: O tema pesquisado
            - articles: Lista de objetos WikipediaArticle com title, content e pageid
            - summary: Um resumo estruturado das informações coletadas
            
            Retorne o resultado em formato JSON válido que possa ser parseado pelo sistema.
            """,
            "expected_output": """
            Um resumo estruturado das informações coletadas, com os principais tópicos e 
            fatos sobre o tema, no formato JSON de um objeto ResearchResult.
            """
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
            "description": """
            Com base nas informações fornecidas pelo Pesquisador, escreva um artigo informativo 
            com no mínimo 300 palavras. O artigo deve ter introdução, desenvolvimento e conclusão. 
            Inclua subtítulos relevantes e organize o conteúdo de forma lógica.
            
            IMPORTANTE: Seu resultado deve ser um objeto WrittenArticle estruturado com os seguintes campos:
            - title: Título do artigo
            - introduction: Texto da introdução
            - sections: Lista de objetos ArticleSection, cada um com title e content
            - conclusion: Texto da conclusão
            - word_count: Número total de palavras do artigo (deve ser ≥ 300)
            
            Retorne o resultado em formato JSON válido que possa ser parseado pelo sistema.
            """,
            "expected_output": """
            Um artigo bem estruturado com no mínimo 300 palavras sobre o tema solicitado,
            no formato JSON de um objeto WrittenArticle.
            """
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
            "description": """
            Revise o artigo fornecido pelo Redator. Verifique a clareza, coesão, gramática e formatação. 
            Faça as correções necessárias e sugira melhorias quando apropriado.
            
            IMPORTANTE: Seu resultado deve ser um objeto EditedArticle estruturado com os seguintes campos:
            - title: Título do artigo revisado
            - introduction: Texto da introdução revisado
            - sections: Lista de objetos ArticleSection revisados, cada um com title e content
            - conclusion: Texto da conclusão revisado
            - word_count: Número total de palavras do artigo revisado
            - improvements: Lista de melhorias realizadas
            
            NÃO retorne apenas "The article has X words". Retorne o artigo completo no formato solicitado.
            
            Retorne o resultado em formato JSON válido que possa ser parseado pelo sistema.
            """,
            "expected_output": """
            O artigo final revisado e pronto para publicação, no formato JSON de um objeto EditedArticle.
            """
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
            # Log início da execução
            print(f"Iniciando execução da crew para o tópico: {topic}")
            print(f"Usando LLM provider: {self.llm_provider}")
            
            result = self.create_crew().kickoff(inputs=inputs)
            
            # Log do resultado para debug
            print(f"Crew result type: {type(result)}")
            print(f"Crew result content: {result}")
            
            # Handle different result types
            if result is None:
                print("ERRO: Resultado da crew é None")
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
                print("Retornando objeto EditedArticle diretamente")
                return result
            elif isinstance(result, dict):
                print("Convertendo dict para EditedArticle")
                # Se dict não contém 'sections', cria uma lista vazia
                if 'sections' not in result:
                    result['sections'] = []
                # Se dict não contém 'improvements', cria uma lista vazia
                if 'improvements' not in result:
                    result['improvements'] = []
                
                # Verifica se sections é uma lista de dicts
                if 'sections' in result and isinstance(result['sections'], list):
                    # Converte cada dicionário para ArticleSection
                    sections = []
                    for section in result['sections']:
                        if isinstance(section, dict):
                            if 'title' in section and 'content' in section:
                                sections.append(ArticleSection(
                                    title=section['title'],
                                    content=section['content']
                                ))
                            else:
                                # Se não tem title ou content, cria seção default
                                sections.append(ArticleSection(
                                    title="Seção",
                                    content="Conteúdo não especificado"
                                ))
                    result['sections'] = sections
                
                # Certifica que todos os campos obrigatórios existem
                required_fields = ['title', 'introduction', 'conclusion', 'word_count']
                for field in required_fields:
                    if field not in result:
                        if field == 'word_count':
                            result[field] = 0
                        else:
                            result[field] = f"Campo {field} não fornecido"
                
                return EditedArticle(**result)
            elif isinstance(result, str):
                print("Resultado é string, tentando processar")
                # Check if the result is just a simple word count message
                if "article has" in result.lower() and "words" in result.lower():
                    print("ERRO: Resultado é apenas uma mensagem de contagem de palavras")
                    # Criar um artigo adequado em vez da mensagem
                    return EditedArticle(
                        title=f"Artigo sobre {topic}",
                        introduction=f"Este é um artigo sobre {topic}.",
                        sections=[
                            ArticleSection(
                                title="Problema de Geração",
                                content="O sistema falhou em gerar um artigo adequado e retornou apenas uma contagem de palavras."
                            )
                        ],
                        conclusion="Por favor, tente novamente com configurações diferentes.",
                        word_count=0,
                        improvements=["Corrigir o problema de geração do artigo"]
                    )
                
                # Try to process string as JSON first
                try:
                    print("Tentando processar string como JSON")
                    data = json.loads(result)
                    
                    # Se data não contém 'sections', cria uma lista vazia
                    if 'sections' not in data:
                        data['sections'] = []
                    # Se data não contém 'improvements', cria uma lista vazia
                    if 'improvements' not in data:
                        data['improvements'] = []
                    
                    # Verifica se sections é uma lista de dicts
                    if 'sections' in data and isinstance(data['sections'], list):
                        # Converte cada dicionário para ArticleSection
                        sections = []
                        for section in data['sections']:
                            if isinstance(section, dict):
                                if 'title' in section and 'content' in section:
                                    sections.append(ArticleSection(
                                        title=section['title'],
                                        content=section['content']
                                    ))
                                else:
                                    # Se não tem title ou content, cria seção default
                                    sections.append(ArticleSection(
                                        title="Seção",
                                        content="Conteúdo não especificado"
                                    ))
                        data['sections'] = sections
                    
                    # Certifica que todos os campos obrigatórios existem
                    required_fields = ['title', 'introduction', 'conclusion', 'word_count']
                    for field in required_fields:
                        if field not in data:
                            if field == 'word_count':
                                data[field] = 0
                            else:
                                data[field] = f"Campo {field} não fornecido"
                    
                    return EditedArticle(**data)
                except json.JSONDecodeError as e:
                    print(f"Erro ao processar JSON: {e}")
                    # If not JSON, try to extract meaningful content
                    if len(result) > 100:  # Se a string é relativamente longa
                        # Tenta extrair partes significativas da string
                        article_content = result
                        
                        # Tenta identificar título, introdução, etc.
                        title = topic
                        introduction = ""
                        conclusion = ""
                        sections = []
                        
                        # Divide por linhas para tentar extrair estrutura
                        lines = result.split('\n')
                        for i, line in enumerate(lines):
                            if i == 0 and line.strip() and not introduction:
                                title = line.strip().replace('#', '').strip()
                            elif i < 3 and not introduction and line.strip():
                                introduction = line.strip()
                            elif i > len(lines) - 3 and not conclusion and line.strip():
                                conclusion = line.strip()
                            elif line.strip().startswith('#') or line.strip().startswith('##'):
                                # Possível título de seção
                                section_title = line.strip().replace('#', '').strip()
                                section_content = ""
                                # Busca o conteúdo da seção
                                j = i + 1
                                while j < len(lines) and not lines[j].strip().startswith('#'):
                                    section_content += lines[j] + "\n"
                                    j += 1
                                sections.append(ArticleSection(
                                    title=section_title,
                                    content=section_content.strip()
                                ))
                        
                        # Se não conseguiu extrair estrutura adequada
                        if not introduction:
                            introduction = "Introdução não identificada no artigo."
                        if not conclusion:
                            conclusion = "Conclusão não identificada no artigo."
                        if not sections:
                            # Divide o conteúdo em parágrafos
                            paragraphs = [p for p in result.split('\n\n') if p.strip()]
                            if paragraphs:
                                if not introduction and len(paragraphs) > 0:
                                    introduction = paragraphs[0]
                                    paragraphs = paragraphs[1:]
                                if not conclusion and len(paragraphs) > 0:
                                    conclusion = paragraphs[-1]
                                    paragraphs = paragraphs[:-1]
                                # Cria seções a partir dos parágrafos restantes
                                for i, paragraph in enumerate(paragraphs):
                                    sections.append(ArticleSection(
                                        title=f"Seção {i+1}",
                                        content=paragraph
                                    ))
                        
                        return EditedArticle(
                            title=title,
                            introduction=introduction,
                            sections=sections,
                            conclusion=conclusion,
                            word_count=len(result.split()),
                            improvements=["Extraído de texto não estruturado"]
                        )
                    else:
                        # Para strings curtas, retorna um artigo básico
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
                print(f"Tipo de resultado não esperado: {type(result)}")
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
            import traceback
            traceback.print_exc()
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