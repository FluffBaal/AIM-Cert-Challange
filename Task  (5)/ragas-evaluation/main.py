#!/usr/bin/env python
"""Main entry point for RAGAS Evaluation Pipeline"""
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.ragas_evaluation.config import config
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Pipeline - Compare Retrieval Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run retrieval evaluation (recommended)
  python main.py evaluate
  
  # Generate golden dataset only
  python main.py generate-golden
  
  # Run quick test
  python main.py test
  
  # Run with custom PDF
  python main.py evaluate --pdf path/to/document.pdf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Evaluate command (default to retrieval metrics)
    eval_parser = subparsers.add_parser('evaluate', help='Run retrieval evaluation')
    eval_parser.add_argument('--pdf', type=str, help='Path to PDF file')
    eval_parser.add_argument('--mixed-metrics', action='store_true', 
                           help='Use mixed metrics (not recommended)')
    eval_parser.add_argument('--quick', action='store_true',
                           help='Quick evaluation with subset of data')
    
    # Generate golden dataset
    golden_parser = subparsers.add_parser('generate-golden', 
                                        help='Generate golden dataset')
    golden_parser.add_argument('--pdf', type=str, help='Path to PDF file')
    golden_parser.add_argument('--size', type=int, default=12,
                             help='Number of questions to generate')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run quick test')
    
    args = parser.parse_args()
    
    # Default to evaluate if no command specified
    if not args.command:
        args.command = 'evaluate'
    
    # Execute command
    if args.command == 'evaluate':
        if args.mixed_metrics:
            console.print("[yellow]Using mixed metrics (not recommended for retrieval comparison)[/yellow]")
            from src.ragas_evaluation.evaluation.evaluator import main as eval_main
            eval_main()
        else:
            console.print("[green]Using retrieval-focused metrics (recommended)[/green]")
            from run_retrieval_evaluation import main as retrieval_main
            retrieval_main()
            
    elif args.command == 'generate-golden':
        from src.ragas_evaluation.data_generation.golden_dataset import generate_golden_dataset
        from src.ragas_evaluation.utils.document_loader import load_pdf
        
        pdf_path = Path(args.pdf) if args.pdf else config.data_dir / "Never split the diff clean.pdf"
        documents = load_pdf(pdf_path)
        generate_golden_dataset(documents, num_samples=args.size)
        
    elif args.command == 'test':
        from test_retrieval_metrics import main as test_main
        test_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
