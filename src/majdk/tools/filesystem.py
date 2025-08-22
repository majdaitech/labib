"""Filesystem tools for reading and writing files"""

from pathlib import Path
from typing import Optional
from ..agent import tool


@tool("fs.read", "Read content from a file")
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
    
    Returns:
        File content as string
    """
    try:
        path = Path(file_path)
        
        # Basic security check - don't read outside current directory
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths in current directory allowed."
        
        if not path.exists():
            return f"Error: File '{file_path}' not found."
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        # Limit file size to 50KB
        if path.stat().st_size > 50 * 1024:
            return "Error: File too large (max 50KB)."
        
        content = path.read_text(encoding=encoding)
        return content
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool("write_file", "Write content to a file")
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
    
    Returns:
        Success message or error
    """
    try:
        path = Path(file_path)
        
        # Basic security check
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths in current directory allowed."
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding=encoding)
        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool("list_files", "List files in a directory")
def list_files(directory: str = ".") -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory path (default: current directory)
    
    Returns:
        List of files and directories
    """
    try:
        path = Path(directory)
        
        # Security check
        if path.is_absolute() or ".." in str(path):
            return "Error: Access denied. Only relative paths allowed."
        
        if not path.exists():
            return f"Error: Directory '{directory}' not found."
        
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                items.append(f"📄 {item.name} ({size} bytes)")
            elif item.is_dir():
                items.append(f"📁 {item.name}/")
        
        if not items:
            return f"Directory '{directory}' is empty."
        
        return f"Contents of '{directory}':\n" + "\n".join(items)
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"
