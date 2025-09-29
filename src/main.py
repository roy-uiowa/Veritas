#!/usr/bin/env python3
"""
Veritas - Policy Analyzer CLI Chat Interface
"""

import argparse
import sys
import os
from src.analyzer import PolicyAnalyzer

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════╗
    ║              VERITAS POLICY ANALYZER         ║
    ║         AI-Powered Policy Assistant          ║
    ╚══════════════════════════════════════════════╝
    """
    print(banner)

def setup_analyzer(policy_file: str) -> PolicyAnalyzer:
    """Initialize the policy analyzer"""
    print("🔄 Initializing Veritas Policy Analyzer...")
    
    if not os.path.exists(policy_file):
        print(f"❌ Error: Policy file '{policy_file}' not found.")
        print("Please make sure the file exists or provide the correct path.")
        sys.exit(1)
    
    try:
        analyzer = PolicyAnalyzer()
        chunks = analyzer.load_policies(policy_file)
        print(f"✅ Loaded {len(chunks)} policy chunks from '{policy_file}'")
        return analyzer
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        sys.exit(1)

def get_search_parameters():
    """Get search parameters from user"""
    print("\n🔍 Search Configuration:")
    print("1. Similarity Search")
    print("2. MMR Search (Maximal Marginal Relevance)")
    
    while True:
        try:
            choice = input("Choose search method (1 or 2): ").strip()
            if choice == "1":
                search_type = "similarity"
                break
            elif choice == "2":
                search_type = "mmr"
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    # Get number of results
    while True:
        try:
            k = input("Number of results to retrieve (default: 4): ").strip()
            if not k:
                k = 4
                break
            k = int(k)
            if k > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    # Get similarity threshold (for similarity search)
    score_threshold = None
    if search_type == "similarity":
        while True:
            try:
                threshold_input = input("Similarity threshold (0.0-1.0, Enter for no threshold): ").strip()
                if not threshold_input:
                    break
                threshold = float(threshold_input)
                if 0.0 <= threshold <= 1.0:
                    score_threshold = threshold
                    break
                else:
                    print("Please enter a value between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    return search_type, k, score_threshold

def chat_loop(analyzer: PolicyAnalyzer):
    """Main chat loop"""
    print("\n💬 Chat with Veritas! Type 'quit' to exit, 'config' to change search settings")
    
    # Default settings
    current_search_type = "similarity"
    current_k = 4
    current_threshold = None
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Veritas!")
                break
            
            elif query.lower() in ['config', 'settings']:
                current_search_type, current_k, current_threshold = get_search_parameters()
                continue
            
            elif not query:
                continue
            
            print("🤔 Analyzing...")
            
            # Get answer
            result = analyzer.ask(
                query=query,
                search_type=current_search_type,
                k=current_k,
                score_threshold=current_threshold,
                include_sources=True
            )
            
            # Display answer
            print(f"\n✅ Answer:")
            print(f"   {result['answer']}")
            
            # Display sources
            if result.get('sources'):
                print(f"\n📚 Sources (using {current_search_type} search):")
                for i, source in enumerate(result['sources'][:3]):  # Show top 3 sources
                    print(f"   {i+1}. {source['content'][:100]}...")
                    if source.get('score'):
                        print(f"      Score: {source['score']:.3f}")
            
            print(f"\n⚙️  Search: {current_search_type} | Results: {current_k}" + 
                  (f" | Threshold: {current_threshold}" if current_threshold else ""))
        
        except KeyboardInterrupt:
            print("\n👋 Thank you for using Veritas!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Veritas Policy Analyzer CLI")
    parser.add_argument(
        "--policy-file", 
        default="../data/companypolicies.txt",
        help="Path to policy document file (default: ../data/companypolicies.txt)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    analyzer = setup_analyzer(args.policy_file)
    chat_loop(analyzer)

if __name__ == "__main__":
    main()
