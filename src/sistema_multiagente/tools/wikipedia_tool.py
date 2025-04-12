from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import requests
import json
from bs4 import BeautifulSoup
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikipediaSearchInput(BaseModel):
    """Input schema for WikipediaSearchTool."""
    query: str = Field(..., description="Termo de busca para artigos Wikipedia.")
    limit: int = Field(5, description="Número máximo de resultados (máximo 10).")
    language: str = Field("pt", description="Código do idioma (pt, en, es, etc).")

class WikipediaContentInput(BaseModel):
    """Input schema for WikipediaContentTool."""
    page_id: str = Field(..., description="ID da página Wikipedia.")
    language: str = Field("pt", description="Código do idioma (pt, en, es, etc).")
    sections: bool = Field(True, description="Incluir seções no resultado.")

class WikipediaSearchResult(BaseModel):
    """Modelo para resultados de busca na Wikipedia"""
    title: str
    pageid: str
    snippet: str

class WikipediaSearchTool(BaseTool):
    """Ferramenta para buscar artigos na Wikipedia"""
    name: str = "Wikipedia Search"
    description: str = (
        "Busca por artigos na Wikipedia relacionados ao termo de pesquisa. "
        "Útil para encontrar informações sobre um tópico específico."
    )
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str, limit: int = 5, language: str = "pt") -> str:
        """
        Executa a busca na Wikipedia.
        
        Args:
            query: Termo de busca
            limit: Número máximo de resultados (máximo 10)
            language: Código do idioma (pt, en, es, etc)
            
        Returns:
            Resultados da busca em formato JSON string
        """
        try:
            # Validar parâmetros
            if limit > 10:
                limit = 10
                
            # Construir URL da API
            base_url = f"https://{language}.wikipedia.org/w/api.php"
            
            # Definir parâmetros
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srwhat": "text",
                "srprop": "snippet",
                "utf8": 1
            }
            
            # Realizar requisição
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Verificar se há resultados
            if "query" in data and "search" in data["query"]:
                results = data["query"]["search"]
                
                # Formatar resultados
                formatted_results = []
                for item in results:
                    result = WikipediaSearchResult(
                        title=item.get("title", ""),
                        pageid=str(item.get("pageid", "")),
                        snippet=self._clean_snippet(item.get("snippet", ""))
                    )
                    formatted_results.append(result.dict())
                
                return json.dumps(formatted_results, ensure_ascii=False)
            else:
                return json.dumps([], ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao buscar na Wikipedia: {str(e)}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def _clean_snippet(self, snippet: str) -> str:
        """Remove HTML tags do snippet."""
        return BeautifulSoup(snippet, "html.parser").get_text()


class WikipediaContentTool(BaseTool):
    """Ferramenta para obter o conteúdo de um artigo na Wikipedia"""
    name: str = "Wikipedia Content"
    description: str = (
        "Obtém o conteúdo completo de um artigo da Wikipedia a partir do ID da página. "
        "Útil para obter informações detalhadas sobre um tópico."
    )
    args_schema: Type[BaseModel] = WikipediaContentInput

    def _run(self, page_id: str, language: str = "pt", sections: bool = True) -> str:
        """
        Obtém o conteúdo de um artigo da Wikipedia.
        
        Args:
            page_id: ID da página Wikipedia
            language: Código do idioma (pt, en, es, etc)
            sections: Incluir seções no resultado
            
        Returns:
            Conteúdo do artigo em formato JSON string
        """
        try:
            # Construir URL da API
            base_url = f"https://{language}.wikipedia.org/w/api.php"
            
            # Definir parâmetros
            params = {
                "action": "parse",
                "format": "json",
                "pageid": page_id,
                "prop": "text|sections",
                "disabletoc": 1,
                "disableeditsection": 1,
                "utf8": 1
            }
            
            # Realizar requisição
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Verificar se há resultados
            if "parse" in data:
                page_data = data["parse"]
                
                # Extrair conteúdo HTML
                html_content = page_data.get("text", {}).get("*", "")
                
                # Remover elementos indesejados e limpar HTML
                clean_content = self._clean_html_content(html_content)
                
                result = {
                    "title": page_data.get("title", ""),
                    "pageid": page_id,
                    "content": clean_content
                }
                
                # Adicionar seções se solicitado
                if sections and "sections" in page_data:
                    result["sections"] = page_data["sections"]
                
                return json.dumps(result, ensure_ascii=False)
            else:
                return json.dumps({"error": "Não foi possível obter o conteúdo da página"}, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao obter conteúdo da Wikipedia: {str(e)}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    def _clean_html_content(self, html: str) -> str:
        """Limpa o conteúdo HTML da Wikipedia."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remover elementos de tabela de referência
        for table in soup.find_all(class_="references"):
            table.decompose()
        
        # Remover notas de rodapé
        for ref in soup.find_all("sup", class_="reference"):
            ref.decompose()
        
        # Remover links de edição
        for edit in soup.find_all(class_="mw-editsection"):
            edit.decompose()
        
        # Remover infocaixas para obter apenas o texto principal
        for infobox in soup.find_all(class_="infobox"):
            infobox.decompose()
        
        # Obter apenas o texto limpo
        clean_text = soup.get_text()
        
        # Remover múltiplas quebras de linha
        import re
        clean_text = re.sub(r'\n+', '\n', clean_text).strip()
        
        return clean_text


# Exemplo de uso das ferramentas
if __name__ == "__main__":
    # Teste da ferramenta de busca
    search_tool = WikipediaSearchTool()
    search_results = search_tool._run("Inteligência Artificial", limit=3)
    print("Resultados da busca:")
    print(search_results)
    
    # Teste da ferramenta de conteúdo (usando o primeiro resultado da busca)
    results = json.loads(search_results)
    if results and len(results) > 0:
        content_tool = WikipediaContentTool()
        page_id = results[0]["pageid"]
        content = content_tool._run(page_id)
        print("\nConteúdo da página:")
        print(content)