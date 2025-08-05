"""Main module entry point for RAGAS evaluation"""
from rich.console import Console
console = Console()

def main():
    """Run retrieval-focused evaluation by default"""
    console.print("[bold]RAGAS Evaluation Pipeline[/bold]")
    console.print("Using retrieval-focused metrics (recommended)\n")
    
    # Import and run retrieval evaluation
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from run_retrieval_evaluation import main as retrieval_main
    retrieval_main()

if __name__ == "__main__":
    main()
