#!/usr/bin/env python3
# filepath: /home/olavo/sistema_multiagente/run_crew.py
import sys
import argparse
from sistema_multiagente.main import cli_run

def main():
    """Main entry point for running the crew from command line"""
    parser = argparse.ArgumentParser(description="Generate articles using CrewAI")
    parser.add_argument("--topic", "-t", type=str, default="Inteligência Artificial", 
                      help="Topic to research and write about")
    
    args = parser.parse_args()
    
    try:
        cli_run(topic=args.topic)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())