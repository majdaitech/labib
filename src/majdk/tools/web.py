"""Web-related tools"""

from typing import Optional
from ..agent import tool


@tool("web.fetch", "Fetch content from a URL")
def fetch(url: str) -> str:
    """
    Fetch content from a web URL.
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        Text content from the URL
    """
    try:
        import requests
        
        headers = {
            'User-Agent': 'MAJD-Agent-Kit/0.1.0'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Return first 2000 characters to avoid token limits
        content = response.text[:2000]
        if len(response.text) > 2000:
            content += "... (truncated)"
        
        return content
    
    except ImportError:
        return "Error: requests library not installed. Run: pip install requests"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


@tool("search_web", "Search the web (mock implementation)")
def search(query: str, num_results: int = 3) -> str:
    """
    Search the web for information (mock implementation).
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Search results
    """
    # This is a mock implementation
    # In a real implementation, you'd integrate with a search API
    return f"Mock search results for '{query}':\n1. Result one\n2. Result two\n3. Result three"
