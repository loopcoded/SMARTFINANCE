# financial_mas_system/agents/risk_compliance_agent.py
from agents.base_agent import BaseAgent
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import llm_query, extract_json_from_llm_output
from typing import Any, Dict, List
import json
import re # For robust parsing

class RiskComplianceAgent(BaseAgent):
    """
    The Risk Compliance Agent monitors the portfolio for risk exposure and
    ensures adherence to predefined compliance rules, posting alerts to the Shared KG.
    It acts as a decentralized monitor and verifier.
    """
    def __init__(self, agent_id: str, shared_kg_manager: SharedKnowledgeGraphManager):
        super().__init__(agent_id, shared_kg_manager, initial_goal="Maintain portfolio risk within acceptable limits, ensure regulatory compliance, and flag potential breaches.")
        self.max_exposure_per_stock = 0.3 # Max 30% of portfolio value in one stock (example rule)
        self.max_daily_loss_percentage = 0.03 # Max 3% daily loss (example rule, from initial portfolio value for the day)
        self.current_portfolio_value_at_start_of_step = 0.0 # To track for daily loss calculation
        print(f"RiskComplianceAgent '{self.agent_id}' initialized with goal: '{self.goal}'.")

    def _extract_numeric_from_kg_response(self, text: str, default: float = 0.0) -> float:
        """Helper to extract a float from a string, typically from KG query results."""
        match = re.search(r'[-+]?\d*\.?\d+', text)
        return float(match.group(0)) if match else default

    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Perceives by querying the Shared KG for current portfolio holdings, cash,
        total value, and recent market price movements to assess risk.
        """
        print(f"Agent '{self.agent_id}': Perceiving portfolio risk and compliance status.")
        
        # Query for comprehensive portfolio status
        portfolio_status_query = self.shared_kg_manager.query_kg("What are the current cash holdings, stock holdings (quantities for AAPL, GOOG, MSFT), and total portfolio value?")
        
        # Query for current market prices of held stocks
        market_price_query = self.shared_kg_manager.query_kg("What are the current prices for AAPL, GOOG, and MSFT?")
        
        # Query for any new trade executions from the Environment or TradingStrategyAgent
        trade_execution_query = self.shared_kg_manager.query_kg("Have any trades been executed recently (e.g., BUY/SELL orders)?")

        # Get initial portfolio value for daily loss calculation
        current_portfolio_value_str = self.shared_kg_manager.query_kg(f"What was the portfolio value at the beginning of today ({simulation_step_info.get('current_date')})?")
        self.current_portfolio_value_at_start_of_step = self._extract_numeric_from_kg_response(current_portfolio_value_str, 0.0)


        self.current_state = {
            "simulation_date": simulation_step_info.get('current_date'),
            "portfolio_status": portfolio_status_query,
            "market_prices": market_price_query,
            "recent_trades": trade_execution_query,
            "current_portfolio_value_at_start_of_step": self.current_portfolio_value_at_start_of_step
        }
        print(f"Agent '{self.agent_id}': Perceived portfolio: {self.current_state['portfolio_status'][:50]}..., Prices: {self.current_state['market_prices'][:50]}...")
    
    def _extract_numeric_from_kg_response(self, text: str, default: float = 0.0) -> float:
        """Helper to extract a float from a string, typically from KG query results."""
        if "Error:" in text or "quota" in text.lower():
            return default # Return default if it's an error message
        match = re.search(r'[-+]?\d*\.?\d+', text)
        return float(match.group(0)) if match else default
    
    def decide(self) -> str:
        """
        Uses its LLM brain to evaluate perceived risk and compliance against predefined rules
        and dynamic market conditions. Decides if any alerts need to be published to the Shared KG.
        """
        print(f"Agent '{self.agent_id}': Deciding on risk and compliance alerts.")
        
        # Extract current portfolio value and holdings from the perceived state
        current_portfolio_value = self._extract_numeric_from_kg_response(
            self.shared_kg_manager.query_kg("What is the current portfolio value?"), 0.0
        )
        current_cash = self._extract_numeric_from_kg_response(
            self.shared_kg_manager.query_kg("What is the current cash in the portfolio?"), 0.0
        )
        
        # Get current holdings and prices to calculate exposure
        holdings_str = self.shared_kg_manager.query_kg("What are the current stock holdings and their quantities (AAPL, GOOG, MSFT)?")
        prices_str = self.shared_kg_manager.query_kg("What are the current prices for AAPL, GOOG, MSFT?")

        current_holdings = {}
        # Parse holdings (simple regex, improve for robustness)
        for symbol in ["AAPL", "GOOG", "MSFT"]:
            match = re.search(rf'{symbol}[^0-9]*(\d+)', holdings_str)
            if match:
                current_holdings[symbol] = int(match.group(1))
            else:
                current_holdings[symbol] = 0

        current_prices = {}
        for symbol in ["AAPL", "GOOG", "MSFT"]:
            match = re.search(rf'{symbol}[^0-9]*(\d+\.?\d*)', prices_str)
            if match:
                current_prices[symbol] = float(match.group(1))
            else:
                current_prices[symbol] = 0.0 # Price not found

        compliance_prompt = (
            f"You are the Risk Compliance Agent. Your goal is to maintain portfolio risk within acceptable limits and ensure regulatory compliance. "\
            f"Current date: {self.current_state['simulation_date']}. \n"\
            f"Portfolio Status: Cash={current_cash:.2f}, Total Value={current_portfolio_value:.2f}, Holdings: {current_holdings}. \n"\
            f"Market Prices: {current_prices}. \n"\
            f"Previous Portfolio Value (start of day): {self.current_portfolio_value_at_start_of_step:.2f}. \n"\
            f"Recent Trades: {self.current_state['recent_trades']}. \n\n"\
            f"Consider these internal rules:\n"\
            f"- Max exposure per single stock: {self.max_exposure_per_stock * 100}%\n"\
            f"- Max daily portfolio loss: {self.max_daily_loss_percentage * 100}%\n"\
            f"Identify any risk breaches or compliance issues based on these rules and the current situation. "\
            f"Also, assess if any recent trades (from TradingStrategyAgent) appear unusually risky.\n"\
            f"**IMPORTANT: Output your assessment as a JSON list of alert objects. If no issues, output an empty list `[]`.**\n"\
            f"Each object must have 'type' (e.g., \"HighExposureRisk\", \"DailyLossExceeded\", \"ComplianceBreach\"), 'entity' (e.g., \"AAPL\", \"Portfolio\"), and 'description' (string explaining the issue).\n"\
            f"Example: `[{{\"type\": \"HighExposureRisk\", \"entity\": \"AAPL\", \"description\": \"AAPL exposure exceeds 30% of portfolio value.\"}}]`\n"\
            f"**IMPORTANT: Output your assessment as a JSON list of alert objects. If no issues, output an empty list `[]`.**\n"
            f"Each object must have 'type' (e.g., \"HighExposureRisk\", \"DailyLossExceeded\", \"ComplianceBreach\"), 'entity' (e.g., \"AAPL\", \"Portfolio\"), and 'description' (string explaining the issue).\n"
            f"ONLY return the JSON array, no other text, no explanations." # Strengthen this line
        )
        
        llm_alerts_json = llm_query(compliance_prompt, model=self.llm_brain).strip()
        print(f"Agent '{self.agent_id}': LLM identified alerts: {llm_alerts_json[:100]}...")
        
        # Add the LLM's raw alert decision string to the Shared KG for others to see (CooperationManagerAgent)
        self.shared_kg_manager.add_fact(self.agent_id, "proposed_risk_alerts", llm_alerts_json, source_agent_id=self.agent_id)
        
        return llm_alerts_json # This string (expected JSON) will be parsed by execute

    def execute(self, alerts_json_str: str) -> Dict[str, Any]:
        """
        Executes by parsing the LLM's alerts JSON string and adding them to the Shared KG.
        """
        print(f"Agent '{self.agent_id}': Executing: Publishing risk/compliance alerts to Shared KG.")
        extracted_json_str = extract_json_from_llm_output(alerts_json_str)
        published_count = 0
        try:
            alerts = json.loads(extracted_json_str)
            if not isinstance(alerts, list):
                print(f"Agent '{self.agent_id}': LLM alert output is not a JSON list: {alerts_json_str[:100]}...")
                return {"observation": "Failed to parse alerts", "reward": 0.0, "terminated": False, "truncated": False, "info": {"alerts_published": 0}}

            for alert in alerts:
                alert_type = alert.get("type", "GeneralAlert")
                entity = alert.get("entity", "System")
                description = alert.get("description", "No description")

                if self.shared_kg_manager.add_fact(f"RiskAlert:{alert_type}", entity, description, source_agent_id=self.agent_id):
                    published_count += 1
                else:
                    print(f"Agent '{self.agent_id}': Failed to add alert to KG: {alert}")
        except json.JSONDecodeError:
            print(f"Agent '{self.agent_id}': LLM alerts output was not valid JSON. Raw: {alerts_json_str[:100]}...")
            # Fallback for non-JSON output (less robust)
            if "risk" in extracted_json_str.lower() or "compliance" in extracted_json_str.lower(): # Use extracted_json_str
                 if self.shared_kg_manager.add_fact("RiskComplianceAlert", "PotentialIssue", extracted_json_str[:100], source_agent_id=self.agent_id):
                    published_count += 1
        except Exception as e:
            print(f"Agent '{self.agent_id}': An unexpected error occurred during alert execution: {e}")
        
        # Reward based on *absence* of critical risks, so often a small positive reward, or negative for identified issues.
        # Here, let's give a small positive reward if no alerts, or negative if critical alerts identified.
        reward = 0.05 if published_count == 0 else -0.2 * published_count
        return {"observation": f"Published {published_count} alerts", "reward": reward, "terminated": False, "truncated": False, "info": {"alerts_published": published_count}}

    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Learns from whether its risk assessments were accurate and timely (e.g., if a breach was prevented).
        """
        print(f"Agent '{self.agent_id}': Learning from risk monitoring. Reward: {reward:.4f}")
        super().learn(observation, reward, terminated, truncated, info) # Call base learn method for history
        
        # LLM-guided learning: Reflect on the effectiveness of risk detection and compliance.
        learning_prompt = (
            f"As the Risk Compliance Agent, I just assessed risk and published {info.get('alerts_published', 0)} alerts. My reward was {reward}. "\
            f"This reward indicates how well I detected and managed risk. "\
            f"What specific lessons can I learn about identifying risk early, enforcing compliance rules more effectively, or assessing the severity of issues? "\
            f"Consider if the TradingStrategyAgent reacted appropriately to my alerts (if that info is in Shared KG)."\
            f"Focus on improving risk detection and compliance enforcement."
        )
        llm_learning_insight = llm_query(learning_prompt, model=self.llm_brain)
        self.personal_kg_store.add_self_fact(self.agent_id, "learned_risk_insight", llm_learning_insight[:100], source="risk_learning")
        print(f"Agent '{self.agent_id}': Learned: {llm_learning_insight[:100]}...")