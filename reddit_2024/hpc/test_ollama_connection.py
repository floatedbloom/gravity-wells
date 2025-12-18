#!/usr/bin/env python3
"""
Ollama Connection Test Script
============================

Simple script to test Ollama server connection and model availability.
Useful for debugging connection issues before running the full bias analysis.

Usage:
    python test_ollama_connection.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path to import create_conf_bias
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from create_conf_bias import check_ollama_connection
    import ollama
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure ollama package is installed: pip install ollama")
    sys.exit(1)

async def test_connection():
    """Test Ollama connection and model availability"""
    print("Testing Ollama connection...")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Node: {os.uname().nodename}")
    print("-" * 50)
    
    try:
        # Test basic connection
        print("1. Testing basic connection...")
        connected, result = await check_ollama_connection()
        
        if connected:
            print("✓ Ollama server is accessible")
            print(f"Available models: {result}")
            
            # Check for specific model
            target_model = 'qwen2.5:7b'
            if target_model in result:
                print(f"✓ Target model '{target_model}' is available")
            else:
                print(f"✗ Target model '{target_model}' not found")
                print("To install: ollama pull qwen2.5:7b")
            
            # Test a simple API call
            print("\n2. Testing simple API call...")
            try:
                client = ollama.AsyncClient()
                test_model = target_model if target_model in result else result[0]
                
                response = await client.chat(
                    model=test_model,
                    messages=[{'role': 'user', 'content': 'Hello, respond with just "OK"'}],
                    stream=False,
                    options=ollama.Options(top_p=0, seed=12345)
                )
                
                print(f"✓ Test API call successful")
                print(f"Model: {test_model}")
                print(f"Response: {response['message']['content']}")
                
            except Exception as e:
                print(f"✗ Test API call failed: {str(e)}")
                return False
                
        else:
            print(f"✗ Cannot connect to Ollama server: {result}")
            print("\nTroubleshooting steps:")
            print("1. Start Ollama server: ollama serve")
            print("2. Check if Ollama is running: ps aux | grep ollama")
            print("3. Test from command line: ollama list")
            print("4. Check network connectivity from compute nodes")
            return False
            
    except Exception as e:
        print(f"✗ Connection test failed: {str(e)}")
        print("\nPossible issues:")
        print("- Ollama server not running")
        print("- Network connectivity problems")
        print("- Firewall blocking access")
        print("- Ollama not installed properly")
        return False
    
    print("\n✓ All tests passed! Ollama is ready for bias analysis.")
    return True

def main():
    """Main function"""
    print("Ollama Connection Test")
    print("=" * 50)
    
    # Run the async test
    success = asyncio.run(test_connection())
    
    print("\n" + "=" * 50)
    if success:
        print("Status: READY - You can proceed with bias analysis")
        sys.exit(0)
    else:
        print("Status: NOT READY - Fix the issues above before running bias analysis")
        sys.exit(1)

if __name__ == "__main__":
    main()
