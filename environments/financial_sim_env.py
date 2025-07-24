# financial_mas_system/environments/financial_sim_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import datetime
import networkx as nx
from typing import Dict, Any, Tuple, Optional, List
import json # For parsing LLM outputs that should be JSON

# Import from your project's modules
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from config import SIM_START_DATE, SIM_END_DATE, SIM_INITIAL_CASH
import requests # For fetching real data (optional, for demo we can mock)
from config import ALPHA_VANTAGE_API_KEY # Assuming you have this configured

class FinancialSimulationEnv(gym.Env):
    """
    A custom Gymnasium environment for a multi-agent financial simulation.
    It simulates a financial market and provides observations, handles actions,\
    and calculates rewards for agents.
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self,
                 shared_kg_manager: SharedKnowledgeGraphManager,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.shared_kg_manager = shared_kg_manager
        self.render_mode = render_mode

        # --- Environment State ---
        self.current_step = 0
        self.start_date = pd.to_datetime(SIM_START_DATE)
        self.end_date = pd.to_datetime(SIM_END_DATE)
        self.current_date = self.start_date
        self.price_history: pd.DataFrame = self._load_or_generate_market_data() # Stocks, indices
        self.portfolio_value: float = SIM_INITIAL_CASH
        self.cash: float = SIM_INITIAL_CASH
        self.holdings: Dict[str, int] = {} # e.g., {'AAPL': 10, 'GOOG': 5}
        self.initial_portfolio_value_at_step_start = SIM_INITIAL_CASH # For reward calculation

        # --- Define Action Space (for conceptual clarity, agents interact via KG) ---
        # The 'action_space' here is primarily for how a single RL agent would interact
        # directly. In our multi-agent system, agents publish to the KG, and the
        # environment reads/executes those. So, it's more conceptual for this setup.
        # Action 0: No major portfolio rebalance
        # Action 1: Initiate a buy signal processing loop
        # Action 2: Initiate a sell signal processing loop
        self.action_space = spaces.Discrete(3) 

        # --- Define Observation Space (for conceptual clarity, agents interact via KG) ---
        # Similarly, the 'observation_space' defines what a single RL agent would receive.
        # In our system, agents primarily 'observe' by querying the Shared KG.
        # This defines the structure of a generic observation vector the environment *could* provide.
        # [current_price_AAPL, current_price_GOOG, cash, portfolio_value, total_volume_AAPL, total_volume_GOOG]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        print(f"Financial Simulation Environment initialized from {SIM_START_DATE} to {SIM_END_DATE}.")
        print(f"Initial cash: {self.cash}")

    def _load_or_generate_market_data(self) -> pd.DataFrame:
        """
        Loads dummy market data for the simulation period.
        For a real system, this would fetch historical data from a database or API.
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Generate more dynamic dummy data
        np.random.seed(42) # for reproducibility
        aapl_prices = 150 + np.cumsum(np.random.randn(len(dates)) * 1.5)
        goog_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 1.0)
        msft_prices = 200 + np.cumsum(np.random.randn(len(dates)) * 2.0)

        data = {
            'AAPL_price': np.maximum(50, aapl_prices), # Ensure prices don't go too low
            'GOOG_price': np.maximum(50, goog_prices),
            'MSFT_price': np.maximum(50, msft_prices),
            'AAPL_volume': np.random.randint(500000, 2000000, len(dates)),
            'GOOG_volume': np.random.randint(300000, 1500000, len(dates)),
            'MSFT_volume': np.random.randint(600000, 2500000, len(dates)),
        }
        df = pd.DataFrame(data, index=dates)
        
        # Optional: Uncomment to try fetching a small amount of real data (be mindful of API limits)
        # You'd need to adapt this to match your simulation date range.
        # if ALPHA_VANTAGE_API_KEY:
        #     try:
        #         print("Attempting to fetch real data from Alpha Vantage (IBM example)...")
        #         symbol = "IBM" 
        #         url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
        #         r = requests.get(url)
        #         r.raise_for_status() 
        #         real_data = r.json()
        #         if "Time Series (Daily)" in real_data:
        #             daily_df = pd.DataFrame.from_dict(real_data["Time Series (Daily)"], orient='index')
        #             daily_df = daily_df.astype(float)
        #             daily_df.index = pd.to_datetime(daily_df.index)
        #             daily_df = daily_df.rename(columns={'4. close': f'{symbol}_price', '5. volume': f'{symbol}_volume'})
        #             daily_df = daily_df[[f'{symbol}_price', f'{symbol}_volume']].sort_index()
        #             
        #             # Merge with dummy data or replace
        #             df = df.drop(columns=[col for col in df.columns if symbol in col], errors='ignore')
        #             df = df.merge(daily_df, left_index=True, right_index=True, how='left')
        #             df = df.fillna(method='ffill').fillna(method='bfill') # Fill NaNs if data gaps
        #             print(f"Loaded real data for {symbol} from Alpha Vantage (partial merge).")
        #         else:
        #             print(f"Alpha Vantage data for {symbol} not found or API limit hit. Using only dummy data.")
        #     except requests.exceptions.RequestException as e:
        #         print(f"Network error fetching Alpha Vantage data: {e}. Using only dummy data.")
        #     except Exception as e:
        #         print(f"Error processing Alpha Vantage data: {e}. Using only dummy data.")

        return df

    def _get_current_observation(self) -> np.ndarray:
        """
        Generates the current observation vector for the environment.
        This provides a simplified, raw view of the market for *this environment's perspective*.
        Agents will augment this by querying the Shared KG for more semantic info.
        """
        # Ensure we have data for the current date. If not, use the last available.
        if self.current_date not in self.price_history.index:
            closest_date = self.price_history.index[self.price_history.index < self.current_date].max()
            if pd.isna(closest_date): # No data before current_date either, use first
                closest_date = self.price_history.index.min()
            current_market_data = self.price_history.loc[closest_date]
        else:
            current_market_data = self.price_history.loc[self.current_date]

        aapl_price = current_market_data.get('AAPL_price', 0.0)
        goog_price = current_market_data.get('GOOG_price', 0.0)
        aapl_volume = current_market_data.get('AAPL_volume', 0.0)
        goog_volume = current_market_data.get('GOOG_volume', 0.0)

        # Construct observation vector: [AAPL_price, GOOG_price, cash, portfolio_value, AAPL_volume, GOOG_volume]
        observation = np.array([
            aapl_price,
            goog_price,
            self.cash,
            self.portfolio_value,
            aapl_volume,
            goog_volume
        ], dtype=np.float32)
        return observation

    def _update_portfolio_value(self) -> None:
        """Calculates the current portfolio value based on current prices and holdings."""
        self.portfolio_value = self.cash
        # Ensure current_date data is available before calculating portfolio value
        if self.current_date not in self.price_history.index:
            current_prices = self.price_history.loc[self.price_history.index.max()] # Use last known prices
        else:
            current_prices = self.price_history.loc[self.current_date]

        for symbol, quantity in self.holdings.items():
            price_col = f'{symbol}_price'
            if price_col in current_prices:
                current_price = current_prices[price_col]
                self.portfolio_value += current_price * quantity
            else:
                print(f"Warning: Price for {symbol} not found for {self.current_date}. Using 0 for calculation.")
                # self.portfolio_value += 0 # Or use a default/last known price

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.
        This is called at the beginning of each episode.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_date = self.start_date
        self.portfolio_value = SIM_INITIAL_CASH
        self.cash = SIM_INITIAL_CASH
        self.holdings = {}
        self.initial_portfolio_value_at_step_start = SIM_INITIAL_CASH # Reset for reward calculation

        # Crucial for Swarm + KG integration: Clear/Reset the Shared KG for a clean episode start.
        # This simulates a new market 'instance' for the agents to learn in.
        # In a long-running production system, the KG would persist and continually evolve.
        self.shared_kg_manager.graph = nx.DiGraph() # Clear the graph using networkx
        self.shared_kg_manager._save_kg() # Save the empty/reset state to disk

        # Add initial facts to the Shared KG, acting as initial environmental 'pheromones'.
        # These facts allow agents to get initial context.
        self.shared_kg_manager.add_fact("System", "has_current_date", str(self.current_date.date()), source_agent_id="Environment")
        self.shared_kg_manager.add_fact("Portfolio", "has_initial_cash", str(self.cash), source_agent_id="Environment")
        self.shared_kg_manager.add_fact("Portfolio", "has_current_value", str(self.portfolio_value), source_agent_id="Environment")
        
        # Add initial market prices to KG
        if self.current_date in self.price_history.index:
            current_prices = self.price_history.loc[self.current_date]
            for col, value in current_prices.items():
                if '_price' in col:
                    symbol = col.replace('_price', '')
                    self.shared_kg_manager.add_fact("Market", f"has_price_{symbol}", str(value), source_agent_id="Environment")
        
        observation = self._get_current_observation()
        info = {"date": str(self.current_date.date()), "cash": self.cash, "portfolio_value": self.portfolio_value, "holdings": self.holdings}
        
        print(f"\n--- Environment Reset. Current Date: {self.current_date.date()}, Cash: {self.cash} ---")
        return observation, info

    def step(self, agent_actions_from_main_loop: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Runs one timestep of the environment's dynamics, processing agent actions.
        In this multi-agent setup, agents mostly communicate by modifying the Shared KG.
        The environment then reads signals from the KG and updates its state.
        """
        self.current_step += 1
        self.current_date += datetime.timedelta(days=1)

        # Check if simulation is over or if we ran out of market data
        terminated = self.current_date > self.end_date or self.current_date > self.price_history.index.max()
        truncated = False 

        # --- Process Agent Actions/Signals from the Shared KG ---
        # TradingStrategyAgent is expected to have placed "trade signals" in the Shared KG.
        # The environment reads these and executes.
        
        trade_signals_str = self.shared_kg_manager.query_kg("Are there any pending trade signals from TradingStrategyAgent? Provide them in JSON format.")
        
        parsed_signals = self._parse_trade_signals(trade_signals_str)
        
        # Execute trades
        trade_executed_this_step = False
        for signal in parsed_signals:
            symbol = signal.get("symbol")
            trade_type = signal.get("type") # 'BUY' or 'SELL'
            quantity = signal.get("quantity")
            
            if symbol and trade_type and quantity and self.current_date in self.price_history.index:
                current_price = self.price_history.loc[self.current_date, f"{symbol}_price"]
                if trade_type == "BUY":
                    cost = current_price * quantity
                    if self.cash >= cost:
                        self.cash -= cost
                        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                        self.shared_kg_manager.add_fact("Trade", "executed_buy_order", f"{quantity} shares of {symbol} at {current_price:.2f}", source_agent_id="Environment")
                        trade_executed_this_step = True
                        print(f"ENV: Executed BUY: {quantity} {symbol} at {current_price:.2f}")
                    else:
                        print(f"ENV: Trade failed (BUY {symbol}): Insufficient cash. Cash: {self.cash:.2f}, Cost: {cost:.2f}")
                        self.shared_kg_manager.add_fact("Trade", "failed_buy_insufficient_cash", f"{quantity} shares of {symbol}", source_agent_id="Environment")
                elif trade_type == "SELL":
                    if self.holdings.get(symbol, 0) >= quantity:
                        revenue = current_price * quantity
                        self.cash += revenue
                        self.holdings[symbol] -= quantity
                        if self.holdings[symbol] == 0:
                            del self.holdings[symbol]
                        self.shared_kg_manager.add_fact("Trade", "executed_sell_order", f"{quantity} shares of {symbol} at {current_price:.2f}", source_agent_id="Environment")
                        trade_executed_this_step = True
                        print(f"ENV: Executed SELL: {quantity} {symbol} at {current_price:.2f}")
                    else:
                        print(f"ENV: Trade failed (SELL {symbol}): Insufficient holdings. Holdings: {self.holdings.get(symbol,0)}, Quantity: {quantity}")
                        self.shared_kg_manager.add_fact("Trade", "failed_sell_insufficient_holdings", f"{quantity} shares of {symbol}", source_agent_id="Environment")
            elif not (self.current_date in self.price_history.index):
                print(f"ENV: Cannot execute trade for {symbol} on {self.current_date.date()}. No market data for this date.")


        # Update portfolio value based on current market prices after trades
        self._update_portfolio_value()

        # --- Calculate Reward ---
        # Reward is based on daily percentage change in portfolio value.
        # This reward is returned by the environment to the main loop, then passed to agent.learn().
        reward = (self.portfolio_value - self.initial_portfolio_value_at_step_start) / self.initial_portfolio_value_at_step_start if self.initial_portfolio_value_at_step_start else 0.0
        self.initial_portfolio_value_at_step_start = self.portfolio_value # Update for next step's calculation

        # --- Update Shared KG with Environment State ('Pheromones') ---
        # The environment itself acts as an agent publishing its state changes.
        self.shared_kg_manager.add_fact("System", "has_current_date", str(self.current_date.date()), source_agent_id="Environment")
        self.shared_kg_manager.add_fact("Portfolio", "has_cash", str(self.cash), source_agent_id="Environment")
        self.shared_kg_manager.add_fact("Portfolio", "has_value", str(self.portfolio_value), source_agent_id="Environment")
        for symbol, qty in self.holdings.items():
            self.shared_kg_manager.add_fact("Portfolio", f"holds_shares_{symbol}", str(qty), source_agent_id="Environment")
            
        # Add current prices to KG (only if available for current date)
        if self.current_date in self.price_history.index:
            current_market_data_for_kg = self.price_history.loc[self.current_date].to_dict()
            for key, value in current_market_data_for_kg.items():
                if '_price' in key:
                    symbol = key.replace('_price', '')
                    self.shared_kg_manager.add_fact("Market", f"has_price_{symbol}", str(value), source_agent_id="Environment")
                if '_volume' in key:
                    symbol = key.replace('_volume', '')
                    self.shared_kg_manager.add_fact("Market", f"has_volume_{symbol}", str(value), source_agent_id="Environment")


        observation = self._get_current_observation() # This is the generic observation for an RL agent
        info = {
            "date": str(self.current_date.date()),
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value,
            "current_market_data": self.price_history.loc[self.current_date].to_dict() if self.current_date in self.price_history.index else {},
            "trade_executed_this_step": trade_executed_this_step,
            "reward_details": f"Daily return: {reward*100:.2f}%"
        }
        
        print(f"ENV: Step {self.current_step} - Date={self.current_date.date()}, Port Value={self.portfolio_value:.2f}, Daily Reward={reward:.4f}")
        return observation, reward, terminated, truncated, info

    def _parse_trade_signals(self, llm_query_result: str) -> List[Dict[str, Any]]:
        """
        Parses trade signals from LLM's query result. Expects JSON output for robustness.
        """
        signals = []
        try:
            # LLM is prompted to output a JSON list of trade objects.
            # Example: [{"symbol": "AAPL", "type": "BUY", "quantity": 10}]
            parsed_json = json.loads(llm_query_result)
            if isinstance(parsed_json, list):
                for item in parsed_json:
                    if all(k in item for k in ["symbol", "type", "quantity"]):
                        signals.append(item)
            else:
                print(f"ENV: LLM trade signal result is not a list: {llm_query_result}")
        except json.JSONDecodeError:
            print(f"ENV: LLM trade signal result is not valid JSON: {llm_query_result[:100]}...")
            # Fallback for non-JSON output (less robust)
            lower_result = llm_query_result.lower()
            if "buy" in lower_result and "aapl" in lower_result:
                signals.append({"symbol": "AAPL", "type": "BUY", "quantity": 1}) # Default to 1 share
            if "sell" in lower_result and "goog" in lower_result:
                signals.append({"symbol": "GOOG", "type": "SELL", "quantity": 1}) # Default to 1 share
        except Exception as e:
            print(f"ENV: Unexpected error parsing trade signals: {e}. Raw: {llm_query_result[:100]}...")
        return signals

    def render(self) -> None:
        """Renders the environment (optional, print statements serve as basic render)."""
        if self.render_mode == "human":
            # Detailed rendering will be handled by print statements in main.py loop
            pass

    def close(self) -> None:
        """Cleans up resources (e.g., closing connections, saving final state)."""
        print("Financial Simulation Environment closed.")
        # Ensure shared KG is saved one last time
        self.shared_kg_manager._save_kg()