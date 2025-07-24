# knowledge_graphs/personal_kg_store.py
import networkx as nx
import os
import datetime
from config import PERSONAL_KG_DIR
from utils.llm_utils import get_llm_model, llm_query
from typing import List, Tuple

class PersonalKnowledgeGraphStore:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.kg_path = os.path.join(PERSONAL_KG_DIR, f"{agent_id}_personal_kg.gml")
        self.graph = self._load_or_create_kg()
        print(f"Agent \'{self.agent_id}\': Personal KG initialized from {self.kg_path}.")

    def _load_or_create_kg(self) -> nx.DiGraph:
        """Loads an existing personal KG or creates a new one for the agent."""
        os.makedirs(PERSONAL_KG_DIR, exist_ok=True)
        if os.path.exists(self.kg_path):
            try:
                return nx.read_gml(self.kg_path)
            except Exception as e:
                print(f"Error loading personal KG for {self.agent_id} from {self.kg_path}: {e}. Creating new one.")
                return nx.DiGraph()
        else:
            return nx.DiGraph()

    def _save_kg(self) -> None:
        """Saves the current state of the personal KG."""
        nx.write_gml(self.graph, self.kg_path)

    def add_self_fact(self, subject: str, predicate: str, obj: str, source: str = "self_reflection") -> None:
        """
        Adds a fact to the agent's personal KG.
        Subjects should often relate to the agent itself (e.g., self.agent_id).
        """
        # Ensure subject and object are strings for consistency in GML
        self.graph.add_edge(str(subject), str(obj), relation=str(predicate), source=str(source), timestamp=str(datetime.datetime.now()))
        # print(f"Agent \'{self.agent_id}\': Added personal fact: ({subject} - {predicate} -> {obj})")
        self._save_kg()

    def reflect(self, current_situation: str, shared_kg_query_result: str = "") -> str:
        """
        Uses LLM to reflect on the agent's actions/state and update its personal KG.
        This is where 'self-understanding' happens, addressing "don't repeat that mistake".
        """
        print(f"Agent \'{self.agent_id}\': Reflecting on current situation...")
        
        personal_facts = self.get_all_facts()
        # Limit personal facts for the prompt to avoid exceeding context window
        personal_facts_summary = "; ".join([f"({s}, {p}, {o})" for s, p, o in personal_facts[-5:]]) # Last 5 facts

        reflection_prompt = (
            f"You are Agent \'{self.agent_id}\'. Your goal is to improve your performance and avoid past mistakes. "\
            f"Based on your recent experiences and internal knowledge, reflecting on:\n"\
            f"- Current Situation Summary: {current_situation}\n"\
            f"- Relevant Shared Market Context (from Shared KG): {shared_kg_query_result}\n"\
            f"- My Recent Personal Learning/Skills (from my Personal KG): {personal_facts_summary}\n\n"\
            "Critically analyze your actions. What did you do well? What challenges did you face? "\
            "**Specifically, identify any errors or sub-optimal decisions you made, the root cause of those mistakes, and a concrete lesson learned to avoid repeating them.** "\
            "If you gain a new, actionable insight about your performance or a specific learning, express it concisely as a single triple (subject, predicate, object) about yourself. "\
            "Subject should be 'self' or your agent_id. Example: ('self', 'learned_to_adjust_for', 'market_volatility')."\
            "Then, provide a general reflection summary. Always start the summary with 'REFLECTION_SUMMARY:'. "
        )
        
        reflection_response = llm_query(reflection_prompt, model=get_llm_model())
        print(f"Agent \'{self.agent_id}\' LLM reflection generated.")
        
        # --- Parsing and Adding Self-Facts ---
        # Look for explicit triple declarations first
        lines = reflection_response.split('\n')
        new_fact_added = False
        for line in lines:
            if "new self-fact:" in line.lower():
                try:
                    # Attempt to extract (subject, predicate, object) from the string
                    # Example: "New self-fact: ('self', 'learned_to_adjust_for', 'market_volatility')"
                    fact_str = line.split("new self-fact:")[1].strip()
                    if fact_str.startswith("('") and fact_str.endswith("')"):
                        parts = fact_str[2:-2].split("','") # Simple split
                        if len(parts) == 3:
                            subject, predicate, obj = parts
                            self.add_self_fact(subject.strip(), predicate.strip(), obj.strip(), source="llm_reflection")
                            new_fact_added = True
                except Exception as e:
                    print(f"Agent \'{self.agent_id}\': Failed to parse explicit self-fact from reflection: {e}")
        
        # Also add the overall reflection summary if present
        summary_start_idx = reflection_response.find('REFLECTION_SUMMARY:')
        if summary_start_idx != -1:
            summary_content = reflection_response[summary_start_idx + len('REFLECTION_SUMMARY:'):].strip()
            if summary_content:
                self.add_self_fact(self.agent_id, "reflects_on", summary_content[:150], source="llm_reflection_summary")
                print(f"Agent \'{self.agent_id}\': Reflection Summary added to PKG: {summary_content[:100]}...")
        
        return reflection_response


    def get_all_facts(self) -> List[Tuple[str, str, str]]:
        """Returns all facts (triples) stored in the personal KG."""
        facts = []
        for u, v, data in self.graph.edges(data=True):
            facts.append((u, data.get('relation', 'knows'), v))
        return facts

    def query_personal_kg(self, query: str) -> str:
        """Queries the agent's personal knowledge graph using LLM for interpretation."""
        personal_facts = self.get_all_facts()
        # Limit personal facts for the prompt to avoid exceeding context window
        personal_facts_summary = "; ".join([f"({s}, {p}, {o})" for s, p, o in personal_facts[-10:]]) # Last 10 facts

        query_prompt = (
            f"You are Agent \'{self.agent_id}\'. Based on your personal knowledge and experience (facts: {personal_facts_summary}), "\
            f"answer the following question about yourself, your skills, or past performance: \'{query}\'. "\
            f"If information is not available in your personal knowledge graph, state that. Be concise."
        )
        return llm_query(query_prompt, model=get_llm_model())

# Example usage (for testing):
if __name__ == "__main__":
    # Ensure PERSONAL_KG_DIR exists and LLM API key is set in .env
    # This block won't run a full simulation but demonstrates KG functionality.
    print("This module is primarily for classes. Run main.py for full simulation.")
    print("Example: An agent reflecting on a mistake:")
    agent_id = "TestAgent1"
    personal_kg = PersonalKnowledgeGraphStore(agent_id)
    
    # Add some initial facts to the personal KG
    personal_kg.add_self_fact(agent_id, "has_skill", "data analysis")
    personal_kg.add_self_fact(agent_id, "last_action_reward", "-0.5")
    personal_kg.add_self_fact(agent_id, "last_action", "risky_trade")
    
    # Simulate a reflection
    current_situation = "My last trade resulted in a significant loss."
    # A dummy shared_kg_query_result
    shared_kg_context = "Market sentiment was unexpectedly negative, leading to price drop."
    
    reflection_output = personal_kg.reflect(current_situation, shared_kg_context)
    print(f"Full Reflection Output: {reflection_output}")
    
    # Query its own KG after reflection
    query_self = "What did I learn from my last mistake?"
    self_knowledge = personal_kg.query_personal_kg(query_self)
    print(f"Querying personal KG: \'{query_self}\'\\nResponse: {self_knowledge}")
    
    # You can now check the .gml file created in knowledge_graphs/personal_kbs/