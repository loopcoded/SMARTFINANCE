# financial_mas_system/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import datetime # Import datetime for timestamping

# Import from your project's modules
from swarms import Agent as SwarmAgent # This is the 'swarms' library you installed
from knowledge_graphs.personal_kg_store import PersonalKnowledgeGraphStore
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import get_llm_model, llm_query 
from config import LLM_MODEL_NAME, DEFAULT_AGENT_GOAL

class BaseAgent(ABC):
    """
    Abstract Base Class for all financial AI agents.
    Each agent has an LLM brain, a personal knowledge graph, and a defined lifecycle.
    """
    def __init__(self, agent_id: str,
                 shared_kg_manager: SharedKnowledgeGraphManager,
                 initial_goal: str = DEFAULT_AGENT_GOAL,
                 **kwargs):
        self.agent_id = agent_id
        self.shared_kg_manager = shared_kg_manager
        self.personal_kg_store = PersonalKnowledgeGraphStore(agent_id=agent_id)
        
        # Initialize the LLM as the agent's brain
        self.llm_brain = get_llm_model() # Uses the utility function to get a configured LLM model

        self.goal = initial_goal # The agent's current high-level goal
        self.current_state: Dict[str, Any] = {} # Internal representation of its observed state
        self.actions_history: list = [] # To keep track of actions taken
        self.rewards_history: list = [] # To keep track of rewards received
        
        print(f"BaseAgent '{self.agent_id}' initialized with LLM '{LLM_MODEL_NAME}' and personal KG.")
        
        # Initialize the 'swarms' Agent (this wraps the core capabilities)
        # We pass the initialized LLM directly.
        self.swarm_agent = SwarmAgent(
            agent_name=self.agent_id,
            llm=self.llm_brain, 
            max_loops=1, # Agents operate in steps within the simulation loop, not infinite loops for this setup
            # You can add more configurations from kwargs if the 'swarms' agent supports them
        )


    @abstractmethod
    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Gathers information from the environment (primarily Shared KG) and internal state.
        Updates self.current_state.
        This represents the 'observation' step in RL.
        """
        pass

    @abstractmethod
    def decide(self) -> Any:
        """
        Uses its LLM brain and personal/shared knowledge to determine the best action.
        This represents the 'policy' in RL.
        Returns the chosen action. The type of action (e.g., string, dict) depends on the agent.
        """
        pass

    @abstractmethod
    def execute(self, action: Any) -> Dict[str, Any]:
        """
        Executes the chosen action, interacting with the environment (simulation)
        or by posting signals to the Shared KG.
        Returns feedback/result of the action. This could be an observation, reward, etc.
        """
        pass

    @abstractmethod
    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Updates internal policies and knowledge based on the outcome of the action.
        This is where MARL happens for the individual agent, using the feedback from the environment.
        """
        self.rewards_history.append(reward)
        self.actions_history.append(info.get("action_details", str(info.get("action", "N/A")))) # Store action details if available
        # In a full MARL setup, you'd feed this into a Stable Baselines3 model's update logic.
        # For this demonstration, we'll primarily use the 'reflect' phase for "learning."
        pass

    def reflect(self, current_situation_summary: str) -> None:
        """
        Uses its LLM brain to reflect on recent experiences, identify mistakes/insights,
        and update its personal KG. This drives the self-improvement and self-awareness novelty.
        """
        print(f"Agent '{self.agent_id}': Initiating self-reflection.")
        
        # Query its own KG for relevant past experiences and learnings
        past_self_facts = self.personal_kg_store.query_personal_kg(f"What are my key skills, past learnings, and common pitfalls to avoid?")
        
        # Combine current situation, shared KG context, and personal history for reflection
        # Agents query the shared KG to get the latest 'world state' for their reflection context
        shared_kg_context_query = f"What is the current market situation and collective state relevant to {self.agent_id}'s goal '{self.goal}'? Include any recent alerts or outcomes."
        shared_kg_context = self.shared_kg_manager.query_kg(shared_kg_context_query)
        
        reflection_input = (
            f"You are Agent '{self.agent_id}'. Your goal is to continuously improve and avoid past errors.\n"
            f"Here is a summary of your recent experience:\n"
            f"  - Current Situation: {current_situation_summary}\n"
            f"  - Shared Market Context: {shared_kg_context}\n"
            f"  - My Past Learnings/Skills (from my Personal KG): {past_self_facts}\n\n"
            f"Critically analyze your recent actions and the outcomes. What went well? What didn't? "\
            f"**Specifically, identify any mistakes, sub-optimal decisions, or missed opportunities. What was the root cause?** "\
            f"Formulate a concrete lesson learned from this experience that will help you avoid repeating this mistake in the future. "\
            f"If you gained a new, actionable insight or updated a skill, state it concisely as a single triple (subject, predicate, object) about yourself. "\
            f"Subject should be 'self' or '{self.agent_id}'.\n"\
            f"Example New Insight Triple: ('self', 'improved_strategy', 'adapting_to_high_volatility').\n"\
            f"Example Mistake Identified: ('{self.agent_id}', 'made_mistake', 'overtrading_in_bear_market').\n"\
            f"Finally, provide a general summary of your reflection starting with 'REFLECTION_SUMMARY:'.\n"
            f"**Your response must contain 'REFLECTION_SUMMARY:'.**"
        )
        
        reflection_output = llm_query(reflection_input, model=self.llm_brain)
        print(f"Agent '{self.agent_id}' LLM reflection output generated (first 100 chars): {reflection_output[:100]}...")
        
        # --- Parsing and Adding Self-Facts from Reflection ---
        lines = reflection_output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "new insight triple:" in line_lower or "example new insight triple:" in line_lower or \
               "mistake identified:" in line_lower or "example mistake identified:" in line_lower:
                try:
                    # Attempt to extract (subject, predicate, object) from the string
                    start_idx = line_lower.find("new insight triple:")
                    if start_idx == -1:
                        start_idx = line_lower.find("example new insight triple:")
                    if start_idx == -1:
                        start_idx = line_lower.find("mistake identified:")
                    if start_idx == -1:
                        start_idx = line_lower.find("example mistake identified:")
                    
                    if start_idx != -1:
                        fact_str = line[start_idx + len("new insight triple:"):].strip() # Use a generic length
                        if fact_str.startswith("('") and fact_str.endswith("')"):
                            parts_str = fact_str[2:-2] # Remove leading/trailing "('" and "')"
                            parts = parts_str.split("','") # Split by "','"
                            if len(parts) == 3:
                                subject, predicate, obj = parts
                                # Ensure the subject is consistent (either 'self' or agent_id)
                                if subject.strip().lower() == 'self':
                                    subject = self.agent_id
                                self.personal_kg_store.add_self_fact(subject.strip(), predicate.strip(), obj.strip(), source="llm_reflection")
                                print(f"Agent '{self.agent_id}': Added new self-fact from reflection: ({subject}, {predicate}, {obj})")
                except Exception as e:
                    print(f"Agent '{self.agent_id}': Failed to parse structured self-fact from line '{line}': {e}")
        
        # Add the overall reflection summary if present
        summary_start_idx = reflection_output.find('REFLECTION_SUMMARY:')
        if summary_start_idx != -1:
            summary_content = reflection_output[summary_start_idx + len('REFLECTION_SUMMARY:'):].strip()
            if summary_content:
                self.personal_kg_store.add_self_fact(self.agent_id, "reflected_summary", summary_content[:150], source="llm_reflection_summary")
                print(f"Agent '{self.agent_id}': Reflection Summary added to PKG: {summary_content[:100]}...")
        
        return reflection_output


    def run_step(self, simulation_step_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes one full cycle of the agent's lifecycle within the simulation.
        Returns the action taken (if any) and info for the environment/main loop.
        """
        print(f"\n--- Agent '{self.agent_id}' Starting Cycle (Date: {simulation_step_info.get('current_date', 'N/A')}) ---")
        
        # 1. Perceive
        self.perceive(simulation_step_info)
        
        # 2. Decide
        action = self.decide()
        
        # 3. Execute
        # The execute method is expected to return the relevant info for learning
        execution_results = self.execute(action)
        observation = execution_results.get("observation")
        reward = execution_results.get("reward", 0.0)
        terminated = execution_results.get("terminated", False)
        truncated = execution_results.get("truncated", False)
        info = execution_results.get("info", {})
        info["action"] = action # Add the action to info for learning context
        
        # 4. Learn
        self.learn(observation, reward, terminated, truncated, info)
        
        # 5. Reflect (periodically or based on triggers, using LLM brain)
        # Reflect every 5 steps or at the end of an episode
        if terminated or (simulation_step_info.get('step', 0) % 5 == 0 and simulation_step_info.get('step', 0) > 0): 
            current_situation_summary = (
                f"Date: {simulation_step_info.get('current_date', 'N/A')}, "
                f"Action taken: '{action}', Reward received: {reward:.4f}, Terminated: {terminated}."
            )
            self.reflect(current_situation_summary)
            
        print(f"--- Agent '{self.agent_id}' Cycle Complete ---")
        return {"agent_id": self.agent_id, "action": action, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}