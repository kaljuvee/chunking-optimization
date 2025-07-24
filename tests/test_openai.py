#!/usr/bin/env python3
"""
Simple OpenAI API Key Test
Tests if the OpenAI API key is working correctly by making a basic API call.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_api():
    """Test OpenAI API key by making a simple completion request."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file or environment")
        return False
    
    print(f"🔑 Found API key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        print("🔧 Testing OpenAI API connection...")
        
        # Make a simple test request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please respond with 'OpenAI API test successful' and nothing else."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        # Check response
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            print(f"✅ OpenAI API test successful!")
            print(f"📝 Response: {content}")
            print(f"🔢 Tokens used: {response.usage.total_tokens}")
            return True
        else:
            print("❌ Error: No response content received")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check if your API key is valid")
        print("2. Ensure you have sufficient credits")
        print("3. Verify your internet connection")
        print("4. Check if the API key has the correct permissions")
        return False

def test_embedding_api():
    """Test OpenAI embedding API by making a simple embedding request."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        print("🔧 Testing OpenAI Embedding API...")
        
        # Make a simple embedding request
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input="This is a test sentence for embedding."
        )
        
        # Check response
        if response.data and response.data[0].embedding:
            embedding_length = len(response.data[0].embedding)
            print(f"✅ OpenAI Embedding API test successful!")
            print(f"📊 Embedding dimensions: {embedding_length}")
            print(f"🔢 Tokens used: {response.usage.total_tokens}")
            return True
        else:
            print("❌ Error: No embedding data received")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenAI Embedding API: {e}")
        return False

def main():
    """Run all OpenAI API tests."""
    print("🧪 OpenAI API Key Test Suite")
    print("=" * 40)
    
    # Test chat completions
    print("\n1. Testing Chat Completions API...")
    chat_success = test_openai_api()
    
    # Test embeddings
    print("\n2. Testing Embeddings API...")
    embedding_success = test_embedding_api()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print(f"   Chat Completions: {'✅ PASS' if chat_success else '❌ FAIL'}")
    print(f"   Embeddings: {'✅ PASS' if embedding_success else '❌ FAIL'}")
    
    if chat_success and embedding_success:
        print("\n🎉 All tests passed! Your OpenAI API key is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check your API key and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 