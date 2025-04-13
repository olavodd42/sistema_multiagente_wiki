#!/usr/bin/env python3
# filepath: src/sistema_multiagente/run_crew.py
import sys
import os
import argparse
from sistema_multiagente.main import cli_run, serve

def main():
    """Main entry point for running the crew from command line"""
    parser = argparse.ArgumentParser(description="Generate articles using CrewAI or start API server")
    
    # Criar subparsers para diferentes comandos
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Comando para gerar artigo
    generate_parser = subparsers.add_parser("generate", help="Generate an article")
    generate_parser.add_argument("--topic", "-t", type=str, default="Inteligência Artificial", 
                               help="Topic to research and write about")
    generate_parser.add_argument("--llm", "-l", type=str, choices=["groq", "gemini", "openai", "local"], 
                               default="local", help="LLM provider to use")

    # Comando para iniciar servidor API
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind the server")
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Verificar variáveis de ambiente API keys
    if args.command == "generate" or not args.command:
        # Use getattr to safely access args.llm with a default value if it doesn't exist
        llm_provider = getattr(args, "llm", "local")  # Changed from "groq" to "local"
        
        if llm_provider == "groq" and not os.getenv("GROQ_API_KEY"):
            print("Error: GROQ_API_KEY environment variable not set")
            return 1
        elif llm_provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY environment variable not set")
            return 1
        elif llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            return 1
    
    # Executar comando apropriado
    try:
        if args.command == "serve":
            print(f"Starting API server on {args.host}:{args.port}...")
            serve(host=args.host, port=args.port)
            return 0
        elif args.command == "generate" or not args.command:
            # Se não houver comando, assumir "generate"
            topic = getattr(args, "topic", "Inteligência Artificial")
            llm_provider = getattr(args, "llm", "local")  # Changed from "groq" to "local"
            
            print(f"Generating article about '{topic}' using {llm_provider}...")
            cli_run(topic=topic, llm_provider=llm_provider)
            return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())