from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai.project import CrewBase, agent, crew, task
import requests
import json

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class WikipediaTools:
    @tool("Buscar na Wikipedia")
    def search_wikipedia(self, query: str) -> str:
        """Busca por artigos na Wikipedia relacionados ao termo de pesquisa"""
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 5
        }
        response = requests.get("https://pt.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        if "query" in data and "search" in data["query"]:
            results = data["query"]["search"]
            return json.dumps([{"title": item["title"], "pageid": item["pageid"]} for item in results])
        return "Nenhum resultado encontrado."

    @tool("Obter conteúdo da Wikipedia")
    def get_wikipedia_content(self, page_id: str) -> str:
        """Obtém o conteúdo de uma página da Wikipedia pelo seu ID"""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True,
            "exsectionformat": "plain"
        }
        response = requests.get("https://pt.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        if "query" in data and "pages" in data["query"]:
            page = data["query"]["pages"][page_id]
            if "extract" in page:
                return page["extract"]
            return "Não foi possível obter o conteúdo."
    
wikipedia_tools = WikipediaTools()

@CrewBase
class SistemaMultiagente():
    """SistemaMultiagente crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[wikipedia_tools.search_wikipedia, wikipedia_tools.get_wikipedia_content],
            allow_delegation=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            verbose=True,
            allow_delegation=True
        )
    
    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config['editor'],
            verbose=True,
            allow_delegation=True
        )

    # Fix the indentation for these task methods - they should be at the same level as agent methods
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def writing_task(self, context=None) -> Task:
        return Task(
            config=self.tasks_config['writing_task'],
            context=context
        )
    
    @task
    def editing_task(self, context=None) -> Task:
        return Task(
            config=self.tasks_config['editing_task'],
            context=context
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SistemaMultiagente crew"""
        # Create tasks as before
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