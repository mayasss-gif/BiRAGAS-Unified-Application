import json
import os
from typing import List, Tuple

class JSONMemory:
    def __init__(self, filepath: str = "agent_memory.json", max_turns: int = 10):
        self.filepath = filepath
        self.max_turns = max_turns
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def add(self, user_input: str, agent_output: str):
        memory = self.load()
        memory.append({"user": user_input, "agent": agent_output})
        # Limit to last N turns
        memory = memory[-self.max_turns:]
        with open(self.filepath, 'w') as f:
            json.dump(memory, f, indent=2)

    def load(self) -> List[dict]:
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def get_context(self) -> str:
        memory = self.load()
        context = ""
        for turn in memory:
            context += f"User: {turn['user']}\nAgent: {turn['agent']}\n"
        return context
