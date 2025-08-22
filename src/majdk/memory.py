"""Memory and logging system"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path


class Memory:
    """Simple memory system with JSONL logging"""
    
    def __init__(self, log_file: str = "agent_logs.jsonl"):
        self.log_file = Path(log_file)
        self.session_id = str(int(time.time()))
        self.steps = []
    
    def log_step(self, step_type: str, content: str, metadata: Dict[str, Any] = None):
        """Log a step in the agent's process"""
        step = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,  # thought, action, observation, final
            "content": content,
            "metadata": metadata or {}
        }
        
        self.steps.append(step)
        
        # Write to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step) + "\n")
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get all steps from current session"""
        return self.steps.copy()
    
    def clear(self):
        """Clear current session steps"""
        self.steps.clear()
        self.session_id = str(int(time.time()))
    
    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load steps from a specific session"""
        if not self.log_file.exists():
            return []
        
        session_steps = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    step = json.loads(line.strip())
                    if step.get("session_id") == session_id:
                        session_steps.append(step)
                except json.JSONDecodeError:
                    continue
        
        return session_steps
