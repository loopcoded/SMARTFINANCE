# financial_mas_system/agents/news_sentiment_agent.py
from agents.base_agent import BaseAgent
from knowledge_graphs.shared_kg_manager import SharedKnowledgeGraphManager
from utils.llm_utils import llm_query , extract_json_from_llm_output
from typing import Any, Dict, List
import datetime
import json

class NewsSentimentAgent(BaseAgent):
    """
    The News Sentiment Agent analyzes financial news articles, extracts sentiment,
    and posts structured sentiment facts to the Shared KG.
    It acts as a decentralized analyzer and publisher of sentiment 'pheromones'.
    """
    def __init__(self, agent_id: str, shared_kg_manager: SharedKnowledgeGraphManager):
        super().__init__(agent_id, shared_kg_manager, initial_goal="Provide timely and accurate news sentiment for relevant financial assets to inform trading and risk decisions.")
        self.tracked_entities = ["AAPL", "GOOG", "MSFT", "Market", "Economy"] # Entities to monitor news for
        # Use a consistent date for mocking news, otherwise it's hard to verify
        self.mock_news_base_date = datetime.date(2024, 1, 15) 
        self.news_mock_data = self._initialize_mock_news()
        print(f"NewsSentimentAgent '{self.agent_id}' initialized with goal: '{self.goal}'.")

    def _initialize_mock_news(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Initializes a static set of mock news articles for demonstration.
        In a real system, this would be dynamic and fetch from real APIs.
        """
        return {
            "AAPL": [
                {"headline": "Apple (AAPL) announces groundbreaking AI chip, stock expected to surge.", "sentiment_raw": "very positive"},
                {"headline": "Analyst warns of potential iPhone sales slowdown for Apple (AAPL).", "sentiment_raw": "negative"},
                {"headline": "Apple Services revenue beats expectations, driving strong Q1 results.", "sentiment_raw": "positive"},
            ],
            "GOOG": [
                {"headline": "Google (GOOG) invests heavily in quantum computing research, future outlook bright.", "sentiment_raw": "positive"},
                {"headline": "Antitrust probe targets Google's ad business, potential regulatory headwinds.", "sentiment_raw": "negative"},
                {"headline": "Google Cloud reports significant growth, strong performance in enterprise sector.", "sentiment_raw": "positive"},
            ],
            "MSFT": [
                {"headline": "Microsoft (MSFT) partners with OpenAI for next-gen AI services.", "sentiment_raw": "positive"},
                {"headline": "Concerns over Windows market share decline impacting Microsoft (MSFT).", "sentiment_raw": "negative"},
                {"headline": "Microsoft Azure revenue soars, cementing cloud leadership.", "sentiment_raw": "positive"},
            ],
            "Market": [
                {"headline": "Global markets show signs of strong recovery amidst easing inflation concerns.", "sentiment_raw": "optimistic"},
                {"headline": "Rising interest rates cause market jitters, investors turn cautious.", "sentiment_raw": "pessimistic"},
            ],
            "Economy": [
                {"headline": "Strong jobs report boosts economic outlook, recession fears recede.", "sentiment_raw": "positive"},
                {"headline": "Unexpected rise in inflation dampens economic growth forecasts.", "sentiment_raw": "negative"},
            ],
        }

    def perceive(self, simulation_step_info: Dict[str, Any]) -> None:
        """
        Perceives by fetching recent (mock) news articles for tracked entities.
        """
        current_sim_date = simulation_step_info.get('current_date', datetime.date.today())
        print(f"Agent '{self.agent_id}': Perceiving financial news for {current_sim_date}.")
        
        current_news_articles = {}
        for entity in self.tracked_entities:
            # In a real system, this would query a news API for news *since* last check.
            # For this mock, we just get the static list for each entity.
            # The LLM will be prompted to find 'relevant' or 'most recent' from this list.
            current_news_articles[entity] = self.news_mock_data.get(entity, [])
        
        self.current_state["recent_news_articles"] = current_news_articles
        self.current_state["current_sim_date"] = current_sim_date
        print(f"Agent '{self.agent_id}': Ready to analyze news for {len(current_news_articles)} entities.")

    def decide(self) -> str:
        """
        Uses its LLM brain to analyze the sentiment of recent news and formulates
        structured facts (sentiment, key events) to be added to the Shared KG.
        """
        print(f"Agent '{self.agent_id}': Deciding on news sentiment facts to publish.")
        
        # Craft a prompt that guides the LLM to analyze sentiment and output structured JSON
        sentiment_analysis_prompt = (
            f"You are a highly skilled Financial News Sentiment Analyst. Your goal is to extract precise sentiment and key events.\n"
            f"Analyze the following recent news articles for {self.current_state['current_sim_date']}:\n"
            f"{json.dumps(self.current_state['recent_news_articles'], indent=2)}\n\n"
            f"For each tracked entity (AAPL, GOOG, MSFT, Market, Economy), extract the overall sentiment (Positive, Negative, Neutral) "\
            f"and a sentiment score (a float from -1.0 to 1.0, where 1.0 is strongly positive, -1.0 is strongly negative). "\
            f"Also, identify any single most impactful 'key_event' mentioned for each entity, or 'General_Market_Trend' for Market/Economy.\n"\
            f"Output your analysis as a JSON list of objects. Each object should have 'entity', 'sentiment', 'score', 'key_event' (or 'market_trend' if applicable).\n"\
            f"Example for Apple: `{{\"entity\": \"AAPL\", \"sentiment\": \"Positive\", \"score\": 0.8, \"key_event\": \"new AI chip\"}}`\n"\
            f"Example for Market: `{{\"entity\": \"Market\", \"sentiment\": \"Neutral\", \"score\": 0.1, \"market_trend\": \"easing inflation concerns\"}}`\n"\
            f"Only provide the JSON array, no other text. **Output ONLY the JSON array.**"
        )
        
        llm_sentiment_json = llm_query(sentiment_analysis_prompt, model=self.llm_brain).strip()
        print(f"Agent '{self.agent_id}': LLM analyzed sentiment (first 100 chars): {llm_sentiment_json[:100]}...")
        return llm_sentiment_json # This string (expected JSON) will be parsed by execute

    def execute(self, sentiment_json_str: str) -> Dict[str, Any]:
        """
        Executes by parsing the LLM's sentiment JSON string and adding structured facts to the Shared KG.
        Each fact published is a 'digital pheromone' representing sentiment insight.
        """
        print(f"Agent '{self.agent_id}': Executing: Publishing sentiment facts to Shared KG.")
        extracted_json_str = extract_json_from_llm_output(sentiment_json_str)
        published_count = 0
        try:
            sentiment_data_list = json.loads(extracted_json_str)
            if not isinstance(sentiment_data_list, list):
                print(f"Agent '{self.agent_id}': LLM did not output a JSON list for sentiment: {sentiment_json_str[:100]}...")
                return {"observation": "Failed to parse sentiment data", "reward": -0.1, "terminated": False, "truncated": False, "info": {"sentiment_facts_published": 0}}
            
            for sentiment_item in sentiment_data_list:
                entity = sentiment_item.get("entity")
                sentiment = sentiment_item.get("sentiment")
                score = sentiment_item.get("score")
                key_event = sentiment_item.get("key_event")
                market_trend = sentiment_item.get("market_trend")

                if entity and sentiment and score is not None:
                    # Add sentiment fact
                    if self.shared_kg_manager.add_fact(entity, "has_sentiment", sentiment, source_agent_id=self.agent_id):
                        published_count += 1
                    if self.shared_kg_manager.add_fact(entity, "sentiment_score", str(score), source_agent_id=self.agent_id):
                        published_count += 1
                    
                    # Add key event/market trend fact
                    if key_event and key_event != "N/A":
                        if self.shared_kg_manager.add_fact(entity, "driven_by_event", key_event, source_agent_id=self.agent_id):
                            published_count += 1
                    elif market_trend and market_trend != "N/A":
                         if self.shared_kg_manager.add_fact(entity, "influenced_by_trend", market_trend, source_agent_id=self.agent_id):
                            published_count += 1

                else:
                    print(f"Agent '{self.agent_id}': Incomplete sentiment data from LLM: {sentiment_item}")

        except json.JSONDecodeError:
            print(f"Agent '{self.agent_id}': LLM sentiment output was not valid JSON. Raw: {extracted_json_str[:100]}...")
            # Fallback if LLM doesn't output JSON
            if "positive" in extracted_json_str.lower() or "negative" in extracted_json_str.lower():
                self.shared_kg_manager.add_fact("Market", "has_general_sentiment", extracted_json_str[:50], source_agent_id=self.agent_id)
                published_count += 1
        except Exception as e:
            print(f"Agent '{self.agent_id}': An unexpected error occurred during sentiment execution: {e}")
        
        # Reward based on number of valid sentiment facts published.
        return {"observation": f"Published {published_count} sentiment facts", "reward": published_count * 0.15, "terminated": False, "truncated": False, "info": {"sentiment_facts_published": published_count}}

    def learn(self, observation: Any, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> None:
        """
        Learns based on the quality or impact of its sentiment analysis.
        (Future: Can be linked to how well trading decisions based on this sentiment performed).
        """
        print(f"Agent '{self.agent_id}': Learning from sentiment analysis. Reward: {reward}")
        super().learn(observation, reward, terminated, truncated, info) # Call base learn method for history
        
        # LLM-guided learning: Reflect on accuracy, relevance, and impact of sentiment provided.
        learning_prompt = (
            f"As the News Sentiment Agent, my reward for publishing {info.get('sentiment_facts_published', 0)} sentiment facts was {reward}. "\
            f"What does this reward indicate about the accuracy or usefulness of my sentiment analysis? "\
            f"Did the sentiment predictions correlate with subsequent market movements or trading outcomes (if that info is in Shared KG)? "\
            f"How can I improve my news processing, sentiment extraction, or key event identification in the future?"\
            f"Focus on improving the predictive value of my sentiment analysis."
        )
        llm_learning_insight = llm_query(learning_prompt, model=self.llm_brain)
        self.personal_kg_store.add_self_fact(self.agent_id, "learned_sentiment_insight", llm_learning_insight[:100], source="sentiment_learning")
        print(f"Agent '{self.agent_id}': Learned: {llm_learning_insight[:100]}...")