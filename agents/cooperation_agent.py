# financial_mas_system/agents/cooperation_agent.py
from agents.base_agent import BaseAgent
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import llm_query # For LLM-based decision making
from typing import Any, Dict

class CooperationManagerAgent(BaseAgent):
    """
    The Cooperation Manager Agent monitors the Shared KG for anomalies, conflicts,
    or critical issues and intervenes only when necessary.
    It embodies the 'minimal central intervention' novelty.
    """
    def __init__(self, agent_id: str, shared_kg_manager: SharedKnowledgeGraphManager):
        super().__init__(agent_id, shared_kg_manager, initial_goal="Maintain overall system stability, coherence, and resolve critical conflicts with minimal intervention.")
        self.intervention_cooldown = 5 # steps after an intervention before it can intervene again
        self.last_intervention_step = -1
        print(f"CooperationManagerAgent '{self.agent_id}' initialized with goal: '{self.goal}'.")

    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Perceives by actively querying the Shared KG for system status,
        agent alerts (e.g., failed verifications, risk breaches), and potential conflicts.
        """
        current_step = simulation_step_info.get('step', 0)
        print(f"Agent '{self.agent_id}': Perceiving system status from Shared KG at step {current_step}.")
        
        # Query for general system health and alerts from other agents/environment
        system_health_query = self.shared_kg_manager.query_kg("What is the current system health, portfolio value, and any active alerts from RiskComplianceAgent?")
        
        # Query for potential conflicts between agents (e.g., conflicting trade signals, or verification failures)
        # Specifically look for 'VerificationFailure' nodes added by SharedKGManager
        conflict_query = self.shared_kg_manager.query_kg("Are there any conflicting signals, unusual agent behaviors, or recent verification failures in the KG?")
        
        self.current_state = {
            "current_step": current_step,
            "system_health_status": system_health_query,
            "potential_conflicts_alerts": conflict_query,
            "last_intervention_step": self.last_intervention_step
        }
        print(f"Agent '{self.agent_id}': Perceived state: {self.current_state['system_health_status'][:50]}... {self.current_state['potential_conflicts_alerts'][:50]}...")

    def decide(self) -> str:
        """
        Uses its LLM brain and personal KG to assess the perceived state and decide
        whether intervention is needed. Prioritizes minimal intervention.
        """
        print(f"Agent '{self.agent_id}': Deciding on intervention based on perceived state.")
        
        if self.current_state["current_step"] - self.last_intervention_step < self.intervention_cooldown:
            # print(f"Agent '{self.agent_id}': On cooldown. No intervention this step.")
            return "NO_INTERVENTION_ON_COOLDOWN"

        decision_prompt = (
            f"You are the Cooperation Manager. Your core goal is to maintain overall system stability and resolve critical conflicts with **minimal intervention**. "\
            f"Only intervene if there's a clear, urgent threat or sustained incoherence.\n"\
            f"Here is the current perceived system status from the Shared Knowledge Graph:\n"\
            f"System Health & Alerts: {self.current_state['system_health_status']}\n"\
            f"Potential Conflicts/Verification Failures: {self.current_state['potential_conflicts_alerts']}\n\n"\
            f"Based on this, do you need to intervene? If YES, describe the intervention action clearly (e.g., 'PAUSE_TRADING_DUE_TO_HIGH_RISK', 'REQUEST_REPORT:TradingStrategyAgent_due_to_losses', 'RESOLVE_CONFLICT:ConflictingSignals'). "\
            f"If NO, state 'NO_INTERVENTION'.\n"\
            f"Your response must start with 'YES_INTERVENE:' or 'NO_INTERVENTION:' followed by the action or reason."
        )
        
        llm_decision = llm_query(decision_prompt, model=self.llm_brain).strip()
        print(f"Agent '{self.agent_id}': LLM Decision: {llm_decision[:100]}...")

        if llm_decision.startswith("YES_INTERVENE:"):
            action = llm_decision.replace("YES_INTERVENE:", "").strip()
            # Add the intervention decision to Shared KG for other agents/system to see
            self.shared_kg_manager.add_fact(self.agent_id, "decided_to_intervene", action, source_agent_id=self.agent_id)
            self.last_intervention_step = self.current_state["current_step"] # Update cooldown
            return action
        else:
            self.shared_kg_manager.add_fact(self.agent_id, "decided_no_intervention", "System stable or not critical", source_agent_id=self.agent_id)
            return "NO_INTERVENTION"

    def execute(self, action: str) -> Dict[str, Any]:
        """
        Executes the chosen intervention action.
        For simulation, this primarily involves posting a new fact to the Shared KG.
        """
        print(f"Agent '{self.agent_id}': Executing action: {action}")
        
        result_info = {"status": "executed", "action_details": action}
        
        if action.startswith("NO_INTERVENTION"):
            # No direct environmental execution, just a status update
            return {"observation": "System Stable", "reward": 0.05, "terminated": False, "truncated": False, "info": result_info} # Small positive reward for stability
        else:
            # Post a critical system-wide alert to the Shared KG
            self.shared_kg_manager.add_fact("SystemAlert", "critical_intervention_needed", action, source_agent_id=self.agent_id)
            print(f"Agent '{self.agent_id}': CRITICAL ALERT: '{action}' posted to Shared KG.")
            # In a real system, this would trigger actual system changes (e.g., pause trading, override agent decision)
            return {"observation": f"Intervention: {action}", "reward": -0.5, "terminated": False, "truncated": False, "info": result_info} # Negative reward for intervention (ideally avoid)

    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Learns from the outcome of its intervention (or non-intervention),
        adjusting its policy for when and how to intervene.
        """
        print(f"Agent '{self.agent_id}': Learning from observation and reward: {reward}")
        super().learn(observation, reward, terminated, truncated, info) # Call base learn method for history
        
        # LLM-guided learning reflection (distinct from general reflect())
        learning_prompt = (
            f"As the Cooperation Manager, I just made a decision (Action: {info.get('action_details', 'N/A')}) and received a reward of {reward}. "\
            f"This reward indicates how effective my decision was in maintaining system stability. "\
            f"What specific lesson can I learn about *when* to intervene (or not to intervene) or *how* to intervene more effectively "\
            f"to achieve my goal of 'Maintain system stability and resolve critical conflicts with minimal intervention'? "\
            f"Consider if the intervention was timely, overzealous, or if non-intervention led to stability."\
            f"Focus on improving my intervention policy."
        )
        llm_learning_insight = llm_query(learning_prompt, model=self.llm_brain)
        self.personal_kg_store.add_self_fact(self.agent_id, "learned_intervention_policy", llm_learning_insight[:100], source="intervention_learning")
        print(f"Agent '{self.agent_id}': Learned: {llm_learning_insight[:100]}...")