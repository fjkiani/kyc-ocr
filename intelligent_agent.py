"""
Customer Insight Agent Implementation
===================================

This module implements an AI-powered customer insight agent that leverages multiple
Azure AI services to analyze customer interactions and provide intelligent recommendations.

Key Features:
------------
1. Real-time sentiment analysis with historical trending
2. Multi-modal theme identification and analysis
3. AI-driven product recommendations
4. Natural language explanation generation

Architecture:
------------
The agent integrates several Azure services:
- Azure Text Analytics: For sentiment analysis and entity recognition
- Azure OpenAI: For embeddings and natural language generation
- Azure Conversation Analysis: For theme detection and intent understanding

Dependencies:
------------
- azure-cognitiveservices-speech
- azure-ai-textanalytics
- azure-ai-language-conversations
- azure-openai
- numpy

Usage Example:
-------------
```python
# Initialize the agent with Azure service credentials
agent = CustomerInsightAgent(
    text_analytics_key="your_key",
    text_analytics_endpoint="your_endpoint",
    openai_key="your_key",
    openai_endpoint="your_endpoint",
    conversation_endpoint="your_endpoint",
    conversation_key="your_key"
)

# Process a customer interaction
async def process_customer_interaction():
    # Example conversation text
    conversation_text = "I'm looking for investment options with good returns but low risk"
    customer_id = "cust_123"

    # Step 1: Analyze sentiment trends
    # This provides both current sentiment and historical trend analysis
    sentiment_data = await agent.analyze_sentiment_trends(conversation_text)
    print(f"Current Sentiment: {sentiment_data['current_sentiment']}")
    print(f"Sentiment Trend: {sentiment_data['historical_trend']['trend']}")

    # Step 2: Identify conversation themes
    # This combines key phrase extraction, entity recognition, and theme analysis
    themes = await agent.identify_key_themes(conversation_text)
    print(f"Identified Themes: {[theme['theme'] for theme in themes]}")

    # Step 3: Generate personalized recommendations
    # Uses themes and sentiment to score and rank products
    recommendations = await agent.recommend_products(
        customer_id,
        themes,
        sentiment_data
    )
    print(f"Top Recommendation: {recommendations[0]['name']}")

    # Step 4: Generate natural language explanation
    # Creates a context-aware explanation of the recommendations
    explanation = await agent.generate_recommendation_explanation(
        recommendations,
        {
            'sentiment': sentiment_data,
            'themes': themes
        }
    )
    print(f"Recommendation Explanation: {explanation}")

# Example response structure:
# {
#     'sentiment_data': {
#         'current_sentiment': 'neutral',
#         'historical_trend': {'trend': 'improving', 'confidence': 0.75},
#         'sentiment_velocity': 0.15
#     },
#     'themes': [
#         {'theme': 'investment', 'type': 'key_phrase', 'confidence': 0.8},
#         {'theme': 'risk management', 'type': 'entity', 'confidence': 0.9}
#     ],
#     'recommendations': [
#         {
#             'product_id': 'prod_1',
#             'name': 'Premium Account',
#             'description': 'Enhanced banking services with dedicated support',
#             'relevance_score': 0.85,
#             'final_score': 0.92
#         }
#     ]
# }

Performance Considerations:
-------------------------
1. Sentiment History:
   - Currently uses in-memory storage
   - For production, implement persistent storage (e.g., Azure Cosmos DB)
   - Consider implementing caching for frequent queries

2. API Calls:
   - Methods are async to handle concurrent requests
   - Implement retry logic for API failures
   - Consider implementing request batching for multiple analyses

3. Scalability:
   - Use connection pooling for database connections
   - Implement rate limiting for API calls
   - Consider implementing caching for embeddings

Error Handling:
--------------
The agent should be wrapped with appropriate error handling:
```python
try:
    sentiment_data = await agent.analyze_sentiment_trends(conversation_text)
except Exception as e:
    logger.error(f"Error analyzing sentiment: {str(e)}")
    # Implement fallback behavior
```

Production Deployment:
--------------------
1. Environment Variables:
   - Store API keys in Azure Key Vault
   - Use managed identities for Azure services
   - Implement proper secret rotation

2. Monitoring:
   - Add logging for API calls and performance metrics
   - Monitor API quotas and rate limits
   - Track sentiment analysis accuracy

3. Security:
   - Implement input validation
   - Sanitize conversation text
   - Implement API authentication

4. Compliance:
   - Ensure PII handling compliance
   - Implement data retention policies
   - Add audit logging for sensitive operations
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.openai import AzureOpenAI

class CustomerInsightAgent:
    """
    A sophisticated AI agent for analyzing customer interactions and providing intelligent recommendations.
    
    This agent combines multiple Azure AI services to:
    1. Analyze customer sentiment and track trends
    2. Identify key conversation themes
    3. Generate personalized product recommendations
    4. Provide natural language explanations
    
    The agent maintains historical context and uses various AI models to provide
    data-driven insights and recommendations.
    """
    
    def __init__(self, 
                 text_analytics_key: str,
                 text_analytics_endpoint: str,
                 openai_key: str,
                 openai_endpoint: str,
                 conversation_endpoint: str,
                 conversation_key: str):
        """
        Initialize the Customer Insight Agent with necessary Azure service credentials.
        
        Args:
            text_analytics_key: API key for Azure Text Analytics
            text_analytics_endpoint: Endpoint URL for Text Analytics
            openai_key: API key for Azure OpenAI
            openai_endpoint: Endpoint URL for Azure OpenAI
            conversation_endpoint: Endpoint URL for Conversation Analysis
            conversation_key: API key for Conversation Analysis
        """
        
        # Initialize Azure Text Analytics for sentiment and key phrase extraction
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=text_analytics_endpoint,
            credential=AzureKeyCredential(text_analytics_key)
        )
        
        # Initialize Azure OpenAI for embeddings and natural language generation
        self.openai_client = AzureOpenAI(
            api_key=openai_key,
            api_version="2023-05-15",
            azure_endpoint=openai_endpoint
        )
        
        # Initialize Conversation Analysis for theme detection
        self.conversation_client = ConversationAnalysisClient(
            conversation_endpoint,
            AzureKeyCredential(conversation_key)
        )
        
        # Initialize in-memory storage for tracking sentiment history
        # In production, this should be replaced with a persistent storage solution
        self.sentiment_history = []
        
    async def analyze_sentiment_trends(self, 
                                     conversation_text: str,
                                     historical_window: int = 30) -> Dict:
        """
        Analyze sentiment trends in customer conversations over time.
        
        This method performs real-time sentiment analysis and maintains a historical
        record to identify trends and changes in customer sentiment.
        
        Args:
            conversation_text: The current conversation text to analyze
            historical_window: Number of days of history to consider for trend analysis
            
        Returns:
            Dictionary containing:
            - current_sentiment: The sentiment of the current conversation
            - current_scores: Detailed sentiment scores
            - historical_trend: Overall sentiment trend analysis
            - sentiment_velocity: Rate of sentiment change
        """
        # Perform sentiment analysis on current conversation
        sentiment_result = self.text_analytics_client.analyze_sentiment(
            documents=[conversation_text]
        )[0]
        
        # Record the sentiment result with timestamp for historical tracking
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'sentiment': sentiment_result.sentiment,
            'scores': {
                'positive': sentiment_result.confidence_scores.positive,
                'neutral': sentiment_result.confidence_scores.neutral,
                'negative': sentiment_result.confidence_scores.negative
            }
        })
        
        # Filter recent sentiments within the specified time window
        recent_sentiments = [s for s in self.sentiment_history 
                           if s['timestamp'] > datetime.now() - timedelta(days=historical_window)]
        
        # Compile comprehensive sentiment analysis results
        sentiment_trends = {
            'current_sentiment': sentiment_result.sentiment,
            'current_scores': sentiment_result.confidence_scores,
            'historical_trend': self._calculate_sentiment_trend(recent_sentiments),
            'sentiment_velocity': self._calculate_sentiment_velocity(recent_sentiments)
        }
        
        return sentiment_trends
    
    async def identify_key_themes(self, conversation_text: str) -> List[Dict]:
        """
        Identify and analyze key themes in customer communications using multiple approaches.
        
        This method combines three different analysis techniques:
        1. Key phrase extraction for topic identification
        2. Entity recognition for specific mention detection
        3. Custom theme categorization for domain-specific analysis
        
        Args:
            conversation_text: The conversation text to analyze
            
        Returns:
            List of identified themes, each containing:
            - theme: The identified theme text
            - type: The type of theme (key_phrase or entity)
            - confidence: Confidence score for the theme
        """
        # Extract key phrases for topic identification
        key_phrases_response = self.text_analytics_client.extract_key_phrases(
            documents=[conversation_text]
        )[0]
        
        # Perform entity recognition for specific mentions
        entities_response = self.text_analytics_client.recognize_entities(
            documents=[conversation_text]
        )[0]
        
        # Perform custom theme analysis using conversation analysis
        theme_analysis = self.conversation_client.analyze_conversation(
            task={
                "kind": "Conversation",
                "analysisInput": {
                    "conversationItem": {
                        "text": conversation_text,
                        "modality": "text",
                        "language": "en"
                    }
                },
                "parameters": {
                    "projectName": "customer_themes",
                    "deploymentName": "production"
                }
            }
        )
        
        themes = []
        # Process and store key phrases
        for phrase in key_phrases_response.key_phrases:
            theme_entry = {
                'theme': phrase,
                'type': 'key_phrase',
                'confidence': 0.8  # Base confidence for key phrases
            }
            themes.append(theme_entry)
            
        # Process and store recognized entities
        for entity in entities_response.entities:
            theme_entry = {
                'theme': entity.text,
                'type': entity.category,
                'confidence': entity.confidence_score
            }
            themes.append(theme_entry)
            
        # Consolidate and deduplicate themes
        return self._consolidate_themes(themes)
    
    async def recommend_products(self, 
                               customer_id: str,
                               conversation_themes: List[Dict],
                               sentiment_data: Dict) -> List[Dict]:
        """
        Generate intelligent product recommendations based on conversation analysis.
        
        This method uses a combination of:
        - Theme-based matching using semantic embeddings
        - Sentiment-aware scoring
        - Historical customer data analysis
        
        Args:
            customer_id: Customer identifier for personalization
            conversation_themes: Identified themes from the conversation
            sentiment_data: Current and historical sentiment analysis
            
        Returns:
            List of recommended products with relevance scores
        """
        # Build comprehensive context for recommendation generation
        context = {
            'customer_id': customer_id,
            'themes': conversation_themes,
            'sentiment': sentiment_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate semantic embeddings for theme matching
        theme_embeddings = await self._generate_embeddings(
            [theme['theme'] for theme in conversation_themes]
        )
        
        # Query product catalog using context and embeddings
        recommendations = await self._query_product_catalog(
            theme_embeddings,
            sentiment_data['current_sentiment'],
            customer_id
        )
        
        # Score and rank recommendations based on multiple factors
        scored_recommendations = self._score_recommendations(
            recommendations,
            sentiment_data,
            conversation_themes
        )
        
        return scored_recommendations
    
    async def generate_recommendation_explanation(self,
                                                recommendations: List[Dict],
                                                customer_context: Dict) -> str:
        """
        Generate natural language explanations for product recommendations.
        
        Uses Azure OpenAI to create personalized, context-aware explanations
        that consider customer sentiment and identified themes.
        
        Args:
            recommendations: List of recommended products
            customer_context: Customer sentiment and conversation context
            
        Returns:
            Natural language explanation of the recommendations
        """
        # Build a context-aware prompt for the explanation
        prompt = self._construct_explanation_prompt(
            recommendations,
            customer_context
        )
        
        # Generate natural language explanation using GPT-4
        response = self.openai_client.chat.completions.create(
            model="gpt-4",  # or your deployed model name
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant providing product recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Balance between creativity and consistency
            max_tokens=300    # Limit response length
        )
        
        return response.choices[0].message.content
    
    def _calculate_sentiment_trend(self, sentiment_history: List[Dict]) -> Dict:
        """
        Calculate overall sentiment trends from historical data using polynomial fitting.
        
        Args:
            sentiment_history: List of historical sentiment records
            
        Returns:
            Dictionary containing trend direction and confidence
        """
        if not sentiment_history:
            return {'trend': 'neutral', 'confidence': 0.0}
            
        # Calculate net sentiment scores and fit trend line
        sentiment_scores = [s['scores']['positive'] - s['scores']['negative'] 
                          for s in sentiment_history]
        trend_direction = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]
        
        # Categorize trend direction with confidence
        return {
            'trend': 'improving' if trend_direction > 0.1 else 'declining' if trend_direction < -0.1 else 'stable',
            'confidence': abs(trend_direction)
        }
    
    def _calculate_sentiment_velocity(self, sentiment_history: List[Dict]) -> float:
        """
        Calculate the rate of sentiment change over recent interactions.
        
        Focuses on the last 5 interactions to identify rapid sentiment shifts.
        
        Args:
            sentiment_history: List of historical sentiment records
            
        Returns:
            Float indicating the rate of sentiment change
        """
        if len(sentiment_history) < 2:
            return 0.0
            
        # Focus on recent interactions for velocity calculation
        recent_sentiments = sentiment_history[-5:]  # Last 5 interactions
        sentiment_changes = [
            s2['scores']['positive'] - s1['scores']['positive']
            for s1, s2 in zip(recent_sentiments[:-1], recent_sentiments[1:])
        ]
        
        return np.mean(sentiment_changes)
    
    def _consolidate_themes(self, themes: List[Dict]) -> List[Dict]:
        """
        Consolidate and deduplicate identified themes while preserving the highest confidence scores.
        
        Args:
            themes: List of raw identified themes
            
        Returns:
            List of unique themes with highest confidence scores
        """
        consolidated = {}
        
        for theme in themes:
            key = theme['theme'].lower()
            if key not in consolidated:
                consolidated[key] = theme
            else:
                # Keep the theme entry with the highest confidence score
                if theme['confidence'] > consolidated[key]['confidence']:
                    consolidated[key] = theme
                    
        return list(consolidated.values())
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate semantic embeddings for text using Azure OpenAI.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",  # or your deployed model
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    async def _query_product_catalog(self,
                                   theme_embeddings: List[List[float]],
                                   sentiment: str,
                                   customer_id: str) -> List[Dict]:
        """
        Query product catalog using semantic matching and customer context.
        
        Note: This is a placeholder implementation. In production, this would
        integrate with your actual product database/catalog system.
        
        Args:
            theme_embeddings: Semantic embeddings of conversation themes
            sentiment: Current customer sentiment
            customer_id: Customer identifier for personalization
            
        Returns:
            List of relevant products with base relevance scores
        """
        # Placeholder implementation - replace with actual product catalog integration
        return [
            {
                'product_id': 'prod_1',
                'name': 'Premium Account',
                'description': 'Enhanced banking services with dedicated support',
                'relevance_score': 0.85
            },
            {
                'product_id': 'prod_2',
                'name': 'Investment Portfolio',
                'description': 'Personalized investment recommendations',
                'relevance_score': 0.75
            }
        ]
    
    def _score_recommendations(self,
                             recommendations: List[Dict],
                             sentiment_data: Dict,
                             themes: List[Dict]) -> List[Dict]:
        """
        Score and rank product recommendations based on multiple factors.
        
        Considers:
        - Base relevance score
        - Customer sentiment
        - Theme alignment
        
        Args:
            recommendations: List of potential product recommendations
            sentiment_data: Customer sentiment analysis
            themes: Identified conversation themes
            
        Returns:
            Sorted list of recommendations with final scores
        """
        scored_recommendations = []
        
        for rec in recommendations:
            # Calculate composite score using multiple factors
            base_score = rec['relevance_score']
            sentiment_modifier = 1.0
            
            # Adjust score based on sentiment
            if sentiment_data['current_sentiment'] == 'positive':
                sentiment_modifier = 1.2  # Boost score for positive sentiment
            elif sentiment_data['current_sentiment'] == 'negative':
                sentiment_modifier = 0.8  # Reduce score for negative sentiment
                
            final_score = base_score * sentiment_modifier
            
            # Create final recommendation entry
            scored_rec = rec.copy()
            scored_rec['final_score'] = final_score
            scored_recommendations.append(scored_rec)
            
        # Sort recommendations by final score
        return sorted(scored_recommendations, key=lambda x: x['final_score'], reverse=True)
    
    def _construct_explanation_prompt(self,
                                    recommendations: List[Dict],
                                    customer_context: Dict) -> str:
        """
        Construct a context-aware prompt for generating recommendation explanations.
        
        Args:
            recommendations: List of recommended products
            customer_context: Customer sentiment and conversation context
            
        Returns:
            Formatted prompt for the language model
        """
        prompt = f"""
        Based on our conversation and your interests, I'd like to recommend some products that might be valuable to you.
        
        Customer Sentiment: {customer_context.get('sentiment', {}).get('current_sentiment')}
        Key Themes: {', '.join([t['theme'] for t in customer_context.get('themes', [])])}
        
        Top Recommendations:
        {self._format_recommendations(recommendations)}
        
        Please generate a natural, conversational explanation for these recommendations,
        focusing on how they address the customer's needs and interests identified in the conversation.
        """
        return prompt
    
    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """
        Format recommendations for inclusion in the explanation prompt.
        
        Args:
            recommendations: List of recommended products
            
        Returns:
            Formatted string of recommendations
        """
        return '\n'.join([
            f"- {rec['name']}: {rec['description']} (Relevance: {rec['relevance_score']:.2f})"
            for rec in recommendations
        ]) 