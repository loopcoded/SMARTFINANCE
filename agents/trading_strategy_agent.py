# financial_mas_system/agents/trading_strategy_agent.py
from agents.base_agent import BaseAgent
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import llm_query
from typing import Any, Dict, List
import json
import re # For robust parsing of numerical values from LLM

class TradingStrategyAgent(BaseAgent):
    """
    The Trading Strategy Agent analyzes market data and news sentiment from the Shared KG
    to formulate trading strategies and propose trade actions. It embodies a MARL agent.
    """
    def __init__(self, agent_id: str, shared_kg_manager: SharedKnowledgeGraphManager):
        super().__init__(agent_id, shared_kg_manager, initial_goal="Generate profitable and risk-adjusted trade signals based on market conditions and sentiment.")
        self.portfolio_targets = {"AAPL": 0.5, "GOOG": 0.5} # Example target allocation (can be dynamic)
        self.min_trade_quantity = 1 # Minimum shares to trade
        print(f"TradingStrategyAgent '{self.agent_id}' initialized with goal: '{self.goal}'.")

    def _extract_numeric_from_kg_response(self, text: str, default: float = 0.0) -> float:
        """Helper to extract a float from a string, typically from KG query results."""
        match = re.search(r'[-+]?\d*\.?\d+', text)
        return float(match.group(0)) if match else default

    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Perceives by querying the Shared KG for current market prices, volumes,
        news sentiment, and current portfolio holdings (from Environment agent's updates).
        """
        print(f"Agent '{self.agent_id}': Perceiving market conditions, sentiment, and portfolio status from Shared KG.")
        
        # Query for current market prices and volumes for tracked symbols
        market_data_query = self.shared_kg_manager.query_kg("What are the current prices and volumes for AAPL, GOOG, and MSFT?")
        
        # Query for news sentiment for tracked entities and overall market/economy
        sentiment_query = self.shared_kg_manager.query_kg("What is the latest sentiment for AAPL, GOOG, MSFT, and the overall Market/Economy (including scores and key events)?")
        
        # Query for current portfolio status (cash, holdings, total value)
        portfolio_status_query = self.shared_kg_manager.query_kg("What is my current cash, stock holdings (quantities for AAPL, GOOG, MSFT), and total portfolio value?")
        
        # Query for any risk alerts from RiskComplianceAgent
        risk_alerts_query = self.shared_kg_manager.query_kg("Are there any active risk or compliance alerts?")

        self.current_state = {
            "simulation_date": simulation_step_info.get('current_date'),
            "market_data": market_data_query,
            "news_sentiment": sentiment_query,
            "portfolio_status": portfolio_status_query,
            "risk_alerts": risk_alerts_query,
            "step_info": simulation_step_info
        }
        print(f"Agent '{self.agent_id}': Perceived state: Prices: {market_data_query[:50]}..., Sentiment: {sentiment_query[:50]}..., Portfolio: {portfolio_status_query[:50]}...")

    def decide(self) -> str:
        """
        Uses its LLM brain to synthesize information, reason about market dynamics,
        and decide on trade actions (BUY/SELL/HOLD). It will output structured trade signals
        as JSON for the environment to execute.
        """
        print(f"Agent '{self.agent_id}': Deciding on trading strategy and actions.")
        
        # Extract numerical values from KG query results for precise reasoning
        current_cash_str = self.shared_kg_manager.query_kg("What is the current cash in the portfolio?")
        current_cash = self._extract_numeric_from_kg_response(current_cash_str, 0.0)

        current_portfolio_value_str = self.shared_kg_manager.query_kg("What is the current portfolio value?")
        current_portfolio_value = self._extract_numeric_from_kg_response(current_portfolio_value_str, 0.0)

        current_aapl_shares_str = self.shared_kg_manager.query_kg("How many shares of AAPL are currently held?")
        current_aapl_shares = int(self._extract_numeric_from_kg_response(current_aapl_shares_str, 0))

        current_goog_shares_str = self.shared_kg_manager.query_kg("How many shares of GOOG are currently held?")
        current_goog_shares = int(self._extract_numeric_from_kg_response(current_goog_shares_str, 0))

        current_msft_shares_str = self.shared_kg_manager.query_kg("How many shares of MSFT are currently held?")
        current_msft_shares = int(self._extract_numeric_from_kg_response(current_msft_shares_str, 0))


        # Incorporate personal knowledge (learned policies, past mistakes) into the decision-making prompt
        personal_strategy_insights = self.personal_kg_store.query_personal_kg(
            "What are my current learned trading strategies, past mistakes to avoid, and risk preferences?"
        )

        decision_prompt = (
            f"You are a Trading Strategy Agent. Your goal is to generate profitable and risk-adjusted trade signals. "\
            f"Current date: {self.current_state['simulation_date']}. "\
            f"Market data (from Shared KG): {self.current_state['market_data']}\n"\
            f"News Sentiment (from Shared KG): {self.current_state['news_sentiment']}\n"\
            f"Portfolio Status (from Shared KG): Cash={current_cash:.2f}, Total Value={current_portfolio_value:.2f}, Holdings: AAPL={current_aapl_shares}, GOOG={current_goog_shares}, MSFT={current_msft_shares}\n"\
            f"Active Risk Alerts (from Shared KG): {self.current_state['risk_alerts']}\n"\
            f"My Personal Strategy Insights (from Personal KG): {personal_strategy_insights}\n\n"\
            f"Based on ALL this information, including current market conditions, sentiment, my portfolio, any risk alerts, and my past learnings, "\
            f"propose specific trade actions (BUY/SELL) for AAPL, GOOG, or MSFT. "\
            f"Consider a simple strategy: if sentiment is positive and risk is low, consider buying. If negative, consider selling or reducing exposure. "\
            f"Adjust quantities based on available cash and risk appetite. Aim for diversified portfolio.\n"\
            f"**IMPORTANT: Output your decision as a JSON list of trade objects. If no trade, output an empty list `[]`.**\n"\
            f"Each object must have 'symbol' (e.g., \"AAPL\"), 'type' (\"BUY\" or \"SELL\"), and 'quantity' (integer > 0).\n"\
            f"Example for Buy: `[{{\"symbol\": \"AAPL\", \"type\": \"BUY\", \"quantity\": 10}}]`\n"\
            f"Example for Sell: `[{{\"symbol\": \"GOOG\", \"type\": \"SELL\", \"quantity\": 5}}]`\n"\
            f"If multiple trades, list them: `[{{\"symbol\": \"AAPL\", \"type\": \"BUY\", \"quantity\": 10}}, {{\"symbol\": \"GOOG\", \"type\": \"SELL\", \"quantity\": 5}}]`\n"\
            f"Ensure quantity is an integer. Consider `min_trade_quantity={self.min_trade_quantity}`.\n"\
            f"ONLY return the JSON array."
        )
        
        llm_trade_decision = llm_query(decision_prompt, model=self.llm_brain).strip()
        print(f"Agent '{self.agent_id}': LLM Proposed Trade Decision: {llm_trade_decision[:100]}...")
        
        # Add the LLM's raw trade decision string to the Shared KG for the environment to pick up
        # This is the 'digital pheromone' representing the trade signal.
        self.shared_kg_manager.add_fact(self.agent_id, "proposed_trade_signal", llm_trade_decision, source_agent_id=self.agent_id)
        
        return llm_trade_decision # Return the raw string for downstream processing/logging

    def execute(self, trade_signals_json_str: str) -> Dict[str, Any]:
        """
        Executes by explicitly confirming the trade signals have been published to the Shared KG.
        The simulation environment will pick up and execute these signals in its step.
        """
        print(f"Agent '{self.agent_id}': Executing: Confirmed trade signals published to Shared KG.")
        
        # In this design, the 'execute' for TradingStrategyAgent mainly confirms it pushed to KG.
        # The actual impact (trade execution) is handled by the environment reading the KG.
        
        return {"observation": f"Proposed trade signals: {trade_signals_json_str}", "reward": 0.0, "terminated": False, "truncated": False, "info": {"proposed_signals": trade_signals_json_str}}

    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Learns from the financial outcome (reward) of its proposed trades,
        adjusting its strategy based on profitability and risk.
        """
        print(f"Agent '{self.agent_id}': Learning from trade outcome. Reward: {reward:.4f}")
        super().learn(observation, reward, terminated, truncated, info) # Call base learn method for history
        
        # LLM-guided learning: Analyze the effectiveness of the proposed trades.
        learning_prompt = (
            f"As the Trading Strategy Agent, I just proposed trades (Signals: {info.get('proposed_signals', 'N/A')}) and received a daily reward of {reward}. "\
            f"This reward reflects the change in portfolio value after trades were potentially executed by the environment. "\
            f"My goal is to generate profitable and risk-adjusted signals. "\
            f"What specific lessons can I learn about my strategy based on this reward and the current market conditions (available in Shared KG)? "\
            f"Should I adjust my approach to 'BUY' or 'SELL' signals, quantity, asset selection, or risk tolerance? "\
            f"Focus on improving my profitability while managing risk."
        )
        llm_learning_insight = llm_query(learning_prompt, model=self.llm_brain)
        self.personal_kg_store.add_self_fact(self.agent_id, "learned_trading_insight", llm_learning_insight[:100], source="trading_learning")
        print(f"Agent '{self.agent_id}': Learned: {llm_learning_insight[:100]}...")