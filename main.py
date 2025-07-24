# financial_mas_system/main.py
import os
import time
import json # For parsing LLM outputs if needed
import datetime # For current date in simulation info

# Import configuration
from config import SIM_INITIAL_CASH, SIM_START_DATE, SIM_END_DATE

# Import Knowledge Graph Managers
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
# PersonalKnowledgeGraphStore is managed by BaseAgent, so no direct import needed here

# Import Environment
from environments.financial_sim_env import FinancialSimulationEnv

# Import Agents
from agents.cooperation_agent import CooperationManagerAgent
from agents.market_data_agent import MarketDataAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.trading_strategy_agent import TradingStrategyAgent
from agents.risk_compliance_agent import RiskComplianceAgent

# Ensure knowledge graph directories exist before running
os.makedirs('knowledge_graphs/personal_kbs', exist_ok=True)
# Ensure the directory for shared KG exists (handled by SharedKnowledgeGraphManager but good practice)
os.makedirs(os.path.dirname(os.path.abspath('knowledge_graphs/shared_financial_kg.gml')), exist_ok=True)


def run_simulation(num_episodes: int = 1, steps_per_episode: int = 30):
    """
    Runs the multi-agent financial simulation, orchestrating agent interactions
    within the environment via the Shared Knowledge Graph.
    """
    print("--- Initializing Multi-Agent Financial System ---")

    # 1. Initialize Shared Knowledge Graph Manager
    # This is the central communication hub (digital pheromone trail)
    shared_kg_manager = SharedKnowledgeGraphManager()

    # 2. Initialize Financial Simulation Environment
    env = FinancialSimulationEnv(shared_kg_manager=shared_kg_manager)

    # 3. Initialize all specialized agents
    # Each agent gets a reference to the shared KG.
    # Note: The order of agents in this list *can* influence immediate behavior
    # if they are executed sequentially in a single step, as one agent's published
    # info might be immediately available to the next in the same step.
    agents = [\
        CooperationManagerAgent(agent_id="CooperationManagerAgent", shared_kg_manager=shared_kg_manager),\
        MarketDataAgent(agent_id="MarketDataAgent", shared_kg_manager=shared_kg_manager),\
        NewsSentimentAgent(agent_id="NewsSentimentAgent", shared_kg_manager=shared_kg_manager),\
        TradingStrategyAgent(agent_id="TradingStrategyAgent", shared_kg_manager=shared_kg_manager),\
        RiskComplianceAgent(agent_id="RiskComplianceAgent", shared_kg_manager=shared_kg_manager),\
    ]

    print(f"\n--- Starting Simulation for {num_episodes} Episodes ---")

    for episode in range(num_episodes):
        print(f"\n### Episode {episode + 1}/{num_episodes} ###")
        
        # Reset the environment at the beginning of each episode
        # This also typically clears/resets the Shared KG for a new clean run
        # The returned 'observation' here is the environment's internal state representation.
        observation, info = env.reset() 
        
        # Initial information for agents to perceive, including current simulation date
        simulation_info_for_agents = {
            "current_date": env.current_date.date(),
            "step": 0,
            "episode": episode + 1
        }

        for step in range(steps_per_episode):
            # Check if simulation needs to terminate early (e.g., ran out of market data)
            if env.current_date > env.price_history.index.max():
                print(f"Simulation ended early: Ran out of market data on {env.current_date.date()}.")
                terminated = True # Set manually if env.step doesn't catch it this way
                break
            
            simulation_info_for_agents["current_date"] = env.current_date.date()
            simulation_info_for_agents["step"] = step + 1

            print(f"\n--- Simulation Step {step + 1}/{steps_per_episode} (Date: {env.current_date.date()}) ---")

            # ------------------------------------------------------------------
            # Multi-Agent Interaction Cycle for this step
            # Agents run their perceive -> decide -> execute -> learn -> reflect cycle.
            # They interact by reading from and writing to the Shared KG (stigmergy).
            # The environment then processes signals from the KG.
            # ------------------------------------------------------------------
            
            # This order ensures sensory agents provide input before decision-makers,
            # and the environment processes based on their combined inputs.
            # CooperationManager acts as an oversight.
            
            # 1. Sensory Agents (Market Data, News Sentiment) perceive and publish to KG
            for agent in [a for a in agents if a.agent_id in ["MarketDataAgent", "NewsSentimentAgent"]]:
                 agent.run_step(simulation_info_for_agents)
                 time.sleep(1) # Small delay to avoid hitting LLM rate limits rapidly

            # 2. Environment processes changes from sensory agents and executes pending trades
            # The environment's `step` method reads directly from the Shared KG for signals.
            # `agent_actions_and_signals` is a dummy here as the env reads from KG.
            current_observation, reward, terminated, truncated, env_info = env.step(agent_actions_from_main_loop={})
            print(f"Environment Updated: Portfolio Value = {env_info['portfolio_value']:.2f}, Daily Reward = {reward:.4f}")
            
            # 3. Decision/Risk Agents (Trading Strategy, Risk Compliance) perceive updated KG and decide
            for agent in [a for a in agents if a.agent_id in ["TradingStrategyAgent", "RiskComplianceAgent"]]:
                agent.run_step(simulation_info_for_agents)
                time.sleep(1) # Small delay

            # 4. Cooperation Manager Agent monitors and potentially intervenes
            for agent in [a for a in agents if a.agent_id == "CooperationManagerAgent"]:
                agent.run_step(simulation_info_for_agents)
                time.sleep(1) # Small delay
            
            # --- End of Agent Interaction Cycle ---

            # Optional: Render environment state
            env.render()

            if terminated:
                print(f"Episode {episode + 1} terminated early on {env.current_date.date()}.")
                break
        
        print(f"\n### Episode {episode + 1} Summary ###")
        print(f"Final Date: {env.current_date.date()}")
        print(f"Final Portfolio Value: {env.portfolio_value:.2f}")
        print(f"Final Cash: {env.cash:.2f}")
        print(f"Final Holdings: {env.holdings}")
        
        # Optionally, query the shared KG for a summary of the episode
        episode_summary_query = shared_kg_manager.query_kg(
            f"Provide a concise summary of the trading activity, key market events, overall sentiment, and risk alerts "
            f"that occurred during Episode {episode + 1} from {env.start_date.date()} to {env.current_date.date()}."
        )
        print(f"\nShared KG Episode Summary:\n{episode_summary_query}")
        
        # Agents can also reflect on the full episode after it ends (optional, already done periodically)
        # for agent in agents:
        #     agent.reflect(f"End of Episode {episode + 1}. Final Portfolio Value: {env.portfolio_value:.2f}")

    env.close() # Close the environment, saving final KG state
    print("\n--- Simulation Complete ---")


if __name__ == "__main__":
    # You can adjust the number of episodes and steps per episode here
    # Start with a small number of steps (e.g., 5-10) to observe interactions
    # before running longer simulations.
    run_simulation(num_episodes=1, steps_per_episode=10) # Run for 1 episode, 10 daily steps
    
    print("\n---------------------------------------------------------------------")
    print("To inspect Knowledge Graphs (after simulation completes):")
    print(f"- Shared KG: {os.path.abspath('knowledge_graphs/shared_financial_kg.gml')}")
    print(f"- Personal KGs in: {os.path.abspath('knowledge_graphs/personal_kbs/')}")
    print("You can open .gml files with graph visualization tools like Gephi (https://gephi.org/).")
    print("---------------------------------------------------------------------")