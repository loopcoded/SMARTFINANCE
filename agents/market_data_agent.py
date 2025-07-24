# financial_mas_system/agents/market_data_agent.py
from agents.base_agent import BaseAgent
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import llm_query, extract_json_from_llm_output
import re
import requests
import json
import datetime
from config import ALPHA_VANTAGE_API_KEY
from typing import Any, Dict, List

class MarketDataAgent(BaseAgent):
    """
    The Market Data Agent is responsible for fetching and fprocessing raw market data
    (e.g., stock prices, volumes) and adding structured facts to the Shared KG.
    It acts as a decentralized sensor for the swarm.
    """
    def __init__(self, agent_id: str, shared_kg_manager: SharedKnowledgeGraphManager):
        super().__init__(agent_id, shared_kg_manager, initial_goal="Provide accurate, timely, and relevant market data insights to the swarm via the Shared KG.")
        self.symbols_to_track = ["AAPL", "GOOG", "MSFT"] # Example symbols
        self.last_fetch_date: Dict[str, datetime.date] = {s: datetime.date(2023,12,31) for s in self.symbols_to_track} # Tracks last fetched date per symbol
        print(f"MarketDataAgent '{self.agent_id}' initialized with goal: '{self.goal}'.")

    def _fetch_real_time_data(self, symbol: str, current_sim_date: datetime.date) -> Dict[str, Any]:
        """
        Mocks or fetches real-time-like daily data for a given symbol, ensuring it's for the current simulation date.
        Note: Alpha Vantage free tier has severe rate limits (5 calls/min, 500 calls/day).
        For robust testing, consider a mocked data source or a paid API for real data.
        """
        # For simplicity, we'll get data from the environment's internal price_history.
        # In a real system, you'd use external APIs here, managing rate limits.
        
        # Query the Shared KG for current market prices from the Environment agent.
        # This simulates reading the most up-to-date market state.
        kg_price_query = self.shared_kg_manager.query_kg(f"What is the current price and volume for {symbol} on {current_sim_date}?")
        
        # LLM to parse current price/volume from KG query result
        parsing_prompt = (
            f"Extract the price and volume for {symbol} from the following text. "
            f"If multiple prices/volumes are mentioned, take the most recent one. "
            f"**Output ONLY a JSON object: {{\"price\": float, \"volume\": float}}. If not found, return an empty JSON object: {{}}.**\n"
            f"Text: \"{kg_price_query}\""
        )
        parsed_data_str = llm_query(parsing_prompt, model=self.llm_brain)
        extracted_json_str = extract_json_from_llm_output(parsed_data_str)
        try:
            parsed_data = json.loads(extracted_json_str)
            
            # --- MODIFICATION STARTS HERE ---
            # Safely get price and volume, defaulting to None if key is missing or value is null/None
            close_price = parsed_data.get('price')
            volume = parsed_data.get('volume')

            if close_price is not None and volume is not None:
                # Attempt to convert to float. If conversion fails, default to 0.0
                try:
                    close_price = float(close_price)
                except (ValueError, TypeError):
                    close_price = 0.0
                    print(f"Agent '{self.agent_id}': Warning: Non-numeric price for {symbol} received: {parsed_data.get('price')}")
                
                try:
                    volume = float(volume) # Convert volume to float too for consistency
                except (ValueError, TypeError):
                    volume = 0.0
                    print(f"Agent '{self.agent_id}': Warning: Non-numeric volume for {symbol} received: {parsed_data.get('volume')}")
                    
                if close_price > 0 and volume > 0: # Ensure valid data
                    print(f"Agent '{self.agent_id}': Parsed {symbol} data from KG: Price={close_price}, Volume={volume}")
                    return {"symbol": symbol, "price": close_price, "volume": volume, "timestamp": str(current_sim_date)}
                else:
                    print(f"Agent '{self.agent_id}': Parsed zero/negative price/volume for {symbol}. Using fallback.")
                    
            else:
                print(f"Agent '{self.agent_id}': Price or Volume missing in LLM response for {symbol}. Using fallback.")
            # --- MODIFICATION ENDS HERE ---

        except json.JSONDecodeError:
            print(f"Agent '{self.agent_id}': Failed to parse JSON from LLM for {symbol}: {extracted_json_str[:50]}...")
        except Exception as e:
            print(f"Agent '{self.agent_id}': Error processing LLM output for {symbol}: {e}. Raw: {extracted_json_str[:50]}...")

        # Fallback to dummy data if KG query or parsing fails or data is invalid
        print(f"Agent '{self.agent_id}': Falling back to dummy data for {symbol}.")
        dummy_price = 100.0 + (self.current_state.get('step_count', 0) % 10) * 0.5 + self.symbols_to_track.index(symbol) * 10
        dummy_volume = 500000 + (self.current_state.get('step_count', 0) % 5) * 10000
        return {"symbol": symbol, "price": dummy_price, "volume": dummy_volume, "timestamp": str(current_sim_date)}


    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Perceives by fetching current market data for tracked symbols
        and updating its internal state.
        """
        current_date = simulation_step_info.get('current_date', datetime.date.today())
        current_step_count = simulation_step_info.get('step', 0)

        print(f"Agent '{self.agent_id}': Perceiving market data for {current_date}.")
        
        current_market_data = {}
        for symbol in self.symbols_to_track:
            # We fetch data for the current simulation date.
            # This ensures we get data that's relevant to the current environment step.
            data = self._fetch_real_time_data(symbol, current_date)
            if data:
                current_market_data[symbol] = data
                self.last_fetch_date[symbol] = current_date # Update last fetch date
        
        self.current_state["market_data"] = current_market_data
        self.current_state["current_sim_date"] = current_date
        self.current_state["step_count"] = current_step_count
        print(f"Agent '{self.agent_id}': Fetched data for {len(current_market_data)} symbols for {current_date}.")

    def decide(self) -> str:
        """
        Decides which key facts to extract from the raw market data and add to the Shared KG.
        Uses LLM to summarize or identify important trends, representing a 'digital pheromone'.
        """
        print(f"Agent '{self.agent_id}': Deciding what market facts to publish to Shared KG.")
        
        # LLM's role: Summarize and extract key facts from raw data for the KG
        data_summary_prompt = (
            f"You are the Market Data Agent. Your goal is to provide accurate, timely, and relevant market data insights. "\
            f"Analyze the following market data for {self.current_state['current_sim_date']}:\n"\
            f"{self.current_state['market_data']}\n\n"\
            f"Identify the most important facts (e.g., 'AAPL price is X', 'GOOG volume is Y', 'overall market trend is Z') "\
            f"that other financial agents (Trading, Risk, etc.) would need. "\
            f"Output these as simple subject-predicate-object triples, one triple per line. "\
            f"Example:\n"\
            f"AAPL has_price 175.50\n"\
            f"AAPL has_volume 1234567\n"\
            f"Market trend is_bullish\n"\
            f"Economy is_stable\n"\
            f"Only provide triples. Do not add any other text. **Output ONLY the triples, one per line.**"
        )
        
        llm_facts_output = llm_query(data_summary_prompt, model=self.llm_brain)
        print(f"Agent '{self.agent_id}': LLM identified facts: {llm_facts_output[:100]}...")
        return llm_facts_output # This string will be parsed by execute

    def execute(self, facts_string: str) -> Dict[str, Any]:
        """
        Executes by parsing the LLM's facts string and adding them to the Shared KG.
        Each fact published is a 'digital pheromone' for other agents.
        """
        print(f"Agent '{self.agent_id}': Executing: Publishing facts to Shared KG.")
        
        lines = facts_string.strip().split('\n')
        published_count = 0
        for line in lines:
            parts = line.split(' ', 2) # Split into subject, predicate, object
            if len(parts) == 3:
                subject, predicate, obj = parts
                # The SharedKGManager's add_fact includes KARMA-like verification
                if self.shared_kg_manager.add_fact(subject, predicate, obj, source_agent_id=self.agent_id):
                    published_count += 1
            else:
                # This could indicate an LLM hallucination or bad formatting.
                print(f"Agent '{self.agent_id}': Could not parse fact line from LLM: '{line}'")
        
        # Reward based on number of valid facts published.
        return {"observation": f"Published {published_count} facts", "reward": published_count * 0.1, "terminated": False, "truncated": False, "info": {"facts_published": published_count}}

    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Learns based on whether the published facts were useful or accurate.
        (Ideally, this would be judged by downstream agents' performance, but for now,
        it's based on internal success of publishing and LLM reflection).
        """
        print(f"Agent '{self.agent_id}': Learning from publishing data. Reward: {reward}")
        super().learn(observation, reward, terminated, truncated, info) # Call base learn method for history
        
        # LLM-guided learning: Reflect on data quality, relevance, and impact.
        learning_prompt = (
            f"As the Market Data Agent, my reward for publishing {info.get('facts_published', 0)} facts was {reward}. "\
            f"What does this reward indicate about the quality or usefulness of the market data I provided to the swarm? "\
            f"How can I improve my data perception, fact extraction, or filtering of irrelevant information in the future? "\
            f"Consider if the data I provided led to good decisions by other agents (if you could infer that from shared KG)."\
            f"Focus on improving data relevance and accuracy."
        )
        llm_learning_insight = llm_query(learning_prompt, model=self.llm_brain)
        self.personal_kg_store.add_self_fact(self.agent_id, "learned_data_quality_insight", llm_learning_insight[:100], source="data_learning")
        print(f"Agent '{self.agent_id}': Learned: {llm_learning_insight[:100]}...")