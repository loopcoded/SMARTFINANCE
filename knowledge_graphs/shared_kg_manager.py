# knowledge_graphs/shared_kg_manager.py
import networkx as nx
import os
from datetime import datetime
from config import SHARED_KG_PATH
from utils.llm_utils import get_llm_model, llm_query # Used for KARMA-like verification
from typing import Dict, Any, List, Tuple

class SharedKnowledgeGraphManager:
    def __init__(self, kg_path: str = SHARED_KG_PATH):
        self.kg_path = kg_path
        self.graph = self._load_or_create_kg()
        print(f"Shared KG initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _load_or_create_kg(self) -> nx.DiGraph:
        """Loads an existing KG or creates a new one."""
        os.makedirs(os.path.dirname(self.kg_path), exist_ok=True)
        if os.path.exists(self.kg_path):
            try:
                return nx.read_gml(self.kg_path)
            except Exception as e:
                print(f"Error loading existing Shared KG from {self.kg_path}: {e}. Creating new one.")
                return nx.DiGraph() # Directed graph for relationships
        else:
            return nx.DiGraph()

    def _save_kg(self) -> None:
        """Saves the current state of the KG."""
        os.makedirs(os.path.dirname(self.kg_path), exist_ok=True)
        nx.write_gml(self.graph, self.kg_path)

    def add_fact(self, subject: str, predicate: str, obj: str, source_agent_id: str = "System") -> bool:
        """
        Adds a new fact (triple) to the shared knowledge graph after a simplified verification.
        Implements a simplified KARMA-like verification using LLM for plausibility.
        Returns True if fact is added, False otherwise.
        """
        # --- Simplified KARMA-like Verification ---
        # In a real KARMA setup, this would involve multiple agents verifying or more sophisticated LLM reasoning.
        # For now, we'll use a simple LLM query for plausibility checking.
        
        # Avoid verifying system/environment updates too strictly unless specifically needed
        if source_agent_id == "Environment" or source_agent_id == "System":
            self.graph.add_edge(subject, obj, relation=predicate, source=source_agent_id, timestamp=str(datetime.now()))
            # print(f"[Shared KG] Fact added directly (Environment/System): ({subject} - {predicate} -> {obj}) by {source_agent_id}")
            self._save_kg()
            return True

        # For agent-generated facts, apply LLM verification
        verification_prompt = (
        f"Given the current general financial context and understanding that agents might provide new observations, "
        f"is the following statement plausible and likely accurate based on common financial knowledge? "
        f"Statement: \'{subject} {predicate} {obj}\'. "
        f"Respond with 'YES' if it seems generally possible, 'NO' if it's highly improbable or contradictory to basic facts. "
        f"Respond with 'YES' or 'NO' only." # Emphasize the strict output
        )
        llm_response = llm_query(verification_prompt, model=get_llm_model()).strip().upper()

        if "YES" in llm_response:
            self.graph.add_edge(subject, obj, relation=predicate, source=source_agent_id, timestamp=str(datetime.now()))
            print(f"[Shared KG] Fact added (verified): ({subject} - {predicate} -> {obj}) by {source_agent_id}")
            self._save_kg()
            return True
        else:
            print(f"[Shared KG] Fact NOT added (verification failed): ({subject} - {predicate} -> {obj}) by {source_agent_id}. LLM said: {llm_response}")
            # Optionally, add a fact about the failed verification for the CooperationManager to see
            self.graph.add_edge(
                "VerificationFailure",
                subject,
                predicate=predicate,
                object=obj,
                source=source_agent_id,
                reason=llm_response,
                timestamp=str(datetime.now())
            )
            for u, v, data in self.graph.edges(data=True):
                print(f"{u} -> {v}, data = {data}")

            self._save_kg()
            return False

    def query_kg(self, query: str) -> str:
        """
        Queries the knowledge graph based on a natural language query using LLM for interpretation.
        This allows agents to 'perceive' the environment and other agents' 'pheromones'.
        """
        # This is a basic LLM-powered query. In reality, you'd convert natural language
        # to more sophisticated graph queries (e.g., Cypher for Neo4j, or NetworkX pathfinding algorithms)
        # and then summarize the results with the LLM.

        # For demonstration, let LLM summarize relevant parts of the KG.
        # We'll pass a simplified representation of the graph (nodes and some recent edges)
        # to keep the prompt within LLM context limits.
        
        nodes_summary = list(self.graph.nodes)[:20] # Limit nodes for prompt
        recent_edges_summary = []
        # Get up to 10 most recent edges, assuming 'timestamp' attribute is added
        sorted_edges = sorted(self.graph.edges(data=True), key=lambda x: x[2].get('timestamp', ''), reverse=True)
        for u, v, data in sorted_edges[:10]: # Limit edges for prompt
            recent_edges_summary.append(f"({u} - {data.get('relation', 'related_to')} -> {v} from {data.get('source', 'Unknown')})")

        kg_summary_prompt = (
            f"You are a financial knowledge expert accessing a shared knowledge graph. "
            f"Based on the following known entities and recent facts in the graph:\n"
            f"Entities: {', '.join(nodes_summary) if nodes_summary else 'None'}\n"
            f"Recent Facts (Subject - Relation -> Object - Source): {'; '.join(recent_edges_summary) if recent_edges_summary else 'None'}\n\n"
            f"Answer the query: \'{query}\'. "
            f"Focus on relationships and facts *present in this graph*. If information is not available, state that explicitly. "
            f"Keep your answer concise and directly address the query."
        )
        return llm_query(kg_summary_prompt, model=get_llm_model())

    def get_all_facts(self) -> List[Tuple[str, str, str]]:
        """Returns all facts (triples) stored in the shared KG."""
        facts = []
        for u, v, data in self.graph.edges(data=True):
            facts.append((u, data.get('relation', 'knows'), v))
        return facts

# Example usage (for testing):
if __name__ == "__main__":
    import datetime
    kg_manager = SharedKnowledgeGraphManager()
    
    print("\\n--- Testing add_fact with verification ---")
    kg_manager.add_fact("AAPL", "has_price", "180.50", source_agent_id="MarketDataAgent")
    kg_manager.add_fact("Market", "has_sentiment", "Positive", source_agent_id="NewsSentimentAgent")
    kg_manager.add_fact("Portfolio", "has_value", "105000", source_agent_id="Environment")
    kg_manager.add_fact("TSLA", "price_jumped_by", "1000%_in_1_minute", source_agent_id="MarketDataAgent") # Should fail verification

    print("\\n--- Testing query_kg ---")
    query1 = "What is the price of AAPL?"
    response1 = kg_manager.query_kg(query1)
    print(f"Query: {query1}\\nResponse: {response1}")

    query2 = "What is the overall market sentiment?"
    response2 = kg_manager.query_kg(query2)
    print(f"Query: {query2}\\nResponse: {response2}")

    query3 = "Tell me about Tesla's recent price changes."
    response3 = kg_manager.query_kg(query3)
    print(f"Query: {query3}\\nResponse: {response3}")

    query4 = "What is the population of Mars?" # Irrelevant query
    response4 = kg_manager.query_kg(query4)
    print(f"Query: {query4}\\nResponse: {response4}")

    # Inspect the raw graph for verification failures or general structure
    print(f"\\nTotal nodes in KG: {kg_manager.graph.number_of_nodes()}")
    print(f"Total edges in KG: {kg_manager.graph.number_of_edges()}")
    # To view the GML file, you can open knowledge_graphs/shared_financial_kg.gml in a text editor
    # or use a tool like Gephi.