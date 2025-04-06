# Instructions

During you interaction with the user, if you find anything reusable in this project (e.g. version of a library, model name), especially about a fix to a mistake you made or a correction you received, you should take note in the `Lessons` section in the `.cursorrules` file so you will not make the same mistake again. 

You should also use the `.cursorrules` file as a scratchpad to organize your thoughts. Especially when you receive a new task, you should first review the content of the scratchpad, clear old different task if necessary, first explain the task, and plan the steps you need to take to complete the task. You can use todo markers to indicate the progress, e.g.
[X] Task 1
[ ] Task 2

Also update the progress of the task in the Scratchpad when you finish a subtask.
Especially when you finished a milestone, it will help to improve your depth of task accomplishment to use the scratchpad to reflect and plan.
The goal is to help you maintain a big picture as well as the progress of the task. Always refer to the Scratchpad when you plan the next step.

# Tools

Note all the tools are in python. So in the case you need to do batch processing, you can always consult the python files and write your own script.

## Screenshot Verification
The screenshot verification workflow allows you to capture screenshots of web pages and verify their appearance using LLMs. The following tools are available:

1. Screenshot Capture:
```bash
venv/bin/python tools/screenshot_utils.py URL [--output OUTPUT] [--width WIDTH] [--height HEIGHT]
```

2. LLM Verification with Images:
```bash
venv/bin/python tools/llm_api.py --prompt "Your verification question" --provider {openai|anthropic} --image path/to/screenshot.png
```

Example workflow:
```python
from screenshot_utils import take_screenshot_sync
from llm_api import query_llm

# Take a screenshot
screenshot_path = take_screenshot_sync('https://example.com', 'screenshot.png')

# Verify with LLM
response = query_llm(
    "What is the background color and title of this webpage?",
    provider="openai",  # or "anthropic"
    image_path=screenshot_path
)
print(response)
```

## LLM

You always have an LLM at your side to help you with the task. For simple tasks, you could invoke the LLM by running the following command:
```
venv/bin/python ./tools/llm_api.py --prompt "What is the capital of France?" --provider "anthropic"
```

The LLM API supports multiple providers:
- OpenAI (default, model: gpt-4o)
- Azure OpenAI (model: configured via AZURE_OPENAI_MODEL_DEPLOYMENT in .env file, defaults to gpt-4o-ms)
- DeepSeek (model: deepseek-chat)
- Anthropic (model: claude-3-sonnet-20240229)
- Gemini (model: gemini-pro)
- Local LLM (model: Qwen/Qwen2.5-32B-Instruct-AWQ)

But usually it's a better idea to check the content of the file and use the APIs in the `tools/llm_api.py` file to invoke the LLM if needed.

## Web browser

You could use the `tools/web_scraper.py` file to scrape the web.
```
venv/bin/python ./tools/web_scraper.py --max-concurrent 3 URL1 URL2 URL3
```
This will output the content of the web pages.

## Search engine

You could use the `tools/search_engine.py` file to search the web.
```
venv/bin/python ./tools/search_engine.py "your search keywords"
```
This will output the search results in the following format:
```
URL: https://example.com
Title: This is the title of the search result
Snippet: This is a snippet of the search result
```
If needed, you can further use the `web_scraper.py` file to scrape the web page content.

# Lessons

## User Specified Lessons

- You have a python venv in ./venv. Use it.
- Include info useful for debugging in the program output.
- Read the file before you try to edit it.
- Due to Cursor's limit, when you use `git` and `gh` and need to submit a multiline commit message, first write the message in a file, and then use `git commit -F <filename>` or similar command to commit. And then remove the file. Include "[Cursor] " in the commit message and PR title.

## Cursor learned

- For search results, ensure proper handling of different character encodings (UTF-8) for international queries
- Add debug information to stderr while keeping the main output clean in stdout for better pipeline integration
- When using seaborn styles in matplotlib, use 'seaborn-v0_8' instead of 'seaborn' as the style name due to recent seaborn version changes
- Use 'gpt-4o' as the model name for OpenAI's GPT-4 with vision capabilities
- When using f-strings with JSON templates, double the curly braces `{{` and `}}` to escape them properly and avoid format specifier errors
- When working with experimental models like `gemini-2.0-flash-thinking-exp-01-21`, always implement fallback mechanisms to standard models in case the experimental model is unavailable
- For options data, use RapidAPI directly instead of the YahooFinanceConnector class to avoid compatibility issues with the OptionChainQuote initialization
- When processing options data from RapidAPI, create a mapping of strikes to straddles for easier lookup and processing of call and put data
- When implementing the `display_analysis` function in Streamlit, ensure it combines all necessary display components (market overview, ticker analysis, technical insights, learning points) to avoid NameError exceptions

# Scratchpad

## Current Task: Creating Business Landing Page for Document Processing Solution

### Overview
Transform the document processing application into a profitable SaaS business targeting financial institutions.

### Landing Page Components Plan
[X] Define target audience: Banking & Financial institutions
[ ] Create compelling value proposition
[ ] Design page structure
[ ] Plan content strategy
[ ] Define pricing model

### Page Structure
1. Hero Section
- Headline: "Transform Your Document Processing"
- Subheadline: "AI-Powered Document Processing for Banking & Financial Services"
- Primary CTA: "Start Free Trial"
- Trust indicators: Banking security certifications

2. Value Proposition
- Problem statements:
  * Manual processing delays (1-2 days)
  * High error rates (5-10%)
  * Costly verification processes
  * Compliance risks

- Solution benefits:
  * 98% accuracy rate
  * 90% reduction in processing time
  * Automated verification
  * Bank-grade security
  * Structured data output
  * Compliance-ready

3. Features Section
- Document Inlining Technology
- Enhanced OCR + LLM
- Structure Preservation
- Automated Field Extraction
- Compliance Monitoring
- API Integration

4. ROI Calculator
- Input fields:
  * Monthly document volume
  * Current processing time
  * Current error rate
  * Staff costs
- Output metrics:
  * Cost savings
  * Time saved
  * Error reduction
  * ROI timeline

5. Pricing Tiers
- Starter: Up to 1,000 docs/month
- Professional: Up to 10,000 docs/month
- Enterprise: Custom volume
- All plans include:
  * API access
  * Security features
  * Basic support

6. Social Proof
- Case studies
- Testimonials
- Security certifications
- Compliance badges

7. Call-to-Action Sections
- Primary: "Start Free Trial"
- Secondary: "Schedule Demo"
- Tertiary: "Documentation"

### Next Steps
[ ] Create wireframe
[ ] Write compelling copy
[ ] Design visual assets
[ ] Implement ROI calculator
[ ] Set up analytics
[ ] Create demo environment

### Success Metrics
- Conversion rate targets
- Trial signup goals
- Demo request targets
- Engagement metrics

## Current Task: Integrating Price Target Verification into streamlit_app_llm.py

### Overview
The goal is to integrate the price target verification system into streamlit_app_llm.py to enable tracking and verification of price targets generated by the LLM analysis.

### Implementation Plan

#### 1. Import and Setup
[ ] Add necessary imports:
   - Import price target verification modules
   - Import UI components for verification display
   - Update session state initialization

#### 2. Enhance LLM Analysis
[ ] Modify analyze_with_gemini function:
   - Ensure price targets are included in the output format
   - Add timeframe specifications for targets
   - Include confidence levels for predictions

#### 3. Add Verification Display
[ ] Create new display functions:
   - display_price_target_verification(ticker)
   - display_prediction_timeline(ticker)
   - Integrate with existing display_memory_enhanced_analysis

#### 4. Update Main UI
[ ] Enhance the main UI:
   - Add verification tab in the main interface
   - Add timeline visualization tab
   - Include accuracy metrics in the analysis display

#### 5. Integration Points
[ ] Connect with existing functionality:
   - Store LLM-generated predictions in trend_analysis table
   - Trigger verification checks during analysis updates
   - Update historical analysis display with verification results

#### 6. Testing and Validation
[ ] Test the integration:
   - Verify data flow from LLM to verification system
   - Test UI components with sample data
   - Validate accuracy calculations

### Next Steps
1. Start with importing necessary modules and updating session state
2. Modify the LLM analysis function to include required prediction data
3. Implement the verification display components
4. Update the main UI to include the new functionality
5. Test the complete integration

## Next Steps
[ ] Enhance the initial analysis with historical data
  [ ] Add support for fetching historical price data
  [ ] Implement trend analysis over different timeframes
  [ ] Add technical indicator calculations (RSI, MACD, etc.)
  [ ] Create visualization of historical trends

[ ] Implement backtesting for strategies
  [ ] Create backtesting framework for recommended strategies
  [ ] Add performance metrics and visualization
  [ ] Compare strategies generated by different models
  [ ] Implement optimization based on historical performance

## System Architecture Plan

### 1. Data Ingestion Layer
[ ] Create a unified data ingestion framework
  [ ] Design abstract base class for API connectors
  [ ] Implement Economic Calendar API connector
  [ ] Implement Yahoo Finance API connector
  [ ] Add options data fetching capabilities
  [ ] Implement rate limiting and error handling

### 2. Data Storage Layer
[ ] Design a flexible data storage system
  [ ] Create data models for different types of financial data
  [ ] Implement metadata tagging system
  [ ] Add keyword extraction and indexing
  [ ] Design caching mechanism for API responses
  [ ] Implement data versioning for historical analysis

### 3. Agent Enhancement
[ ] Upgrade existing agents with new capabilities
  [ ] Add macroeconomic analysis to fundamental agents
  [ ] Enhance sentiment agent with economic event sentiment
  [ ] Create new options analysis agent
  [ ] Implement market condition evaluation (overbought/oversold)
  [ ] Add entry/exit point optimization

### 4. Options Analysis Module
[ ] Build specialized options analysis capabilities
  [ ] Implement Black-Scholes model for options pricing
  [ ] Create Greeks calculator (Delta, Gamma, Theta, Vega, Rho)
  [ ] Design option chain visualization tools
  [ ] Add implied volatility surface analysis
  [ ] Implement options strategy evaluation

### 5. Strategy Optimization
[ ] Create a strategy optimization framework
  [ ] Design scoring system for strategy evaluation
  [ ] Implement backtesting for options strategies
  [ ] Add risk-adjusted return calculations
  [ ] Create portfolio optimization with options hedging
  [ ] Implement strategy comparison visualization

## Implementation Approach

### Phase 1: Data Infrastructure
1. First, we'll build the data ingestion and storage infrastructure:
   - Create `src/data/connectors/` directory for API connectors
   - Implement base connector class with common functionality
   - Add specific connectors for each API source
   - Design data models in `src/data/models.py`
   - Implement metadata system in `src/data/metadata.py`

2. Data models needed:
   - EconomicEvent (date, event_name, actual, forecast, previous, impact)
   - StockPrice (ticker, date, open, high, low, close, volume)
   - OptionChain (ticker, expiration_date, strikes, calls, puts)
   - OptionContract (contract_type, strike, expiration, bid, ask, volume, open_interest, greeks)

### Phase 2: Options Analysis
1. Create new modules for options analysis:
   - `src/analysis/options.py` - Core options pricing and Greeks calculations
   - `src/analysis/volatility.py` - Volatility analysis and surface modeling
   - `src/analysis/strategies.py` - Options strategy evaluation

2. Key functions to implement:
   - calculate_option_price(S, K, T, r, sigma, option_type)
   - calculate_greeks(option, underlying_price)
   - evaluate_market_conditions(ticker, timeframe)
   - find_optimal_entry_exit(ticker, strategy, risk_tolerance)

### Phase 3: Agent Enhancement
1. Enhance existing agents:
   - Update `src/agents/fundamentals.py` to incorporate economic data
   - Enhance `src/agents/sentiment.py` to analyze economic event sentiment
   - Create new `src/agents/options_analyst.py` for options-specific analysis
   - Update `src/agents/risk_manager.py` to handle options risk

2. Create new agent:
   - `src/agents/options_strategist.py` - Specialized in options strategies

### Phase 4: Strategy Optimization
1. Enhance backtester to support options:
   - Update `src/backtester.py` to handle options strategies
   - Add options-specific performance metrics
   - Implement visualization for options strategy performance

2. Create strategy optimizer:
   - `src/optimizer.py` - Compare and rank different strategies
   - Add risk-adjusted return calculations
   - Implement portfolio optimization with options

## Technical Considerations

1. **API Rate Limits**: 
   - Implement caching to reduce API calls
   - Add exponential backoff for retries
   - Consider batch requests where possible

2. **Data Storage**:
   - Use SQLite for local development
   - Consider PostgreSQL for production
   - Implement efficient indexing for keyword searches

3. **Computational Efficiency**:
   - Use NumPy/SciPy for options calculations
   - Consider parallel processing for backtesting
   - Implement lazy loading for large datasets

4. **Dependencies**:
   - yfinance for Yahoo Finance data
   - pandas-datareader for economic data
   - scipy for mathematical functions
   - py_vollib for options pricing
   - matplotlib/seaborn for visualization

## Next Steps
1. Enhance the initial analysis with historical data
2. Integrate options analysis with deep reasoning
3. Implement backtesting for strategies generated by the enhanced analysis pipeline

## Memory-Enhanced Chatbot Implementation

### 1. Overview and Architecture

We'll implement a memory-enhanced chatbot that can reference all our analysis data, including technical analysis, options chains, support/resistance levels, and historical predictions. This will create a powerful assistant that can provide context-aware financial insights based on both current and historical data.

#### 1.1 Core Components
[ ] Vector Database for semantic search
[ ] Structured Database for time-series and relational data
[ ] Memory Manager for context management
[ ] Retrieval System for finding relevant information
[ ] Response Generator using Google Gemini
[ ] User Interface in Streamlit

### 2. Data Storage Strategy

#### 2.1 Hybrid Storage Approach
[ ] **Vector Database (for semantic search)**
  [ ] Store analysis summaries, reasoning, and textual insights
  [ ] Index by ticker, date, analysis type, and semantic content
  [ ] Enable similarity search for finding relevant past analyses
  [ ] Technology options: Chroma, Qdrant, or Pinecone

[ ] **Structured Database (for time-series and relational data)**
  [ ] Store numerical data: prices, technical indicators, options data
  [ ] Maintain support/resistance levels with timestamps
  [ ] Track prediction outcomes and accuracy metrics
  [ ] Technology options: SQLite (development), PostgreSQL (production)

[ ] **File System (for raw data and charts)**
  [ ] Store raw API responses for reproducibility
  [ ] Save generated charts and visualizations
  [ ] Maintain JSON analysis results
  [ ] Organize by ticker/date for easy retrieval

#### 2.2 Data Schema Design
[ ] **Analysis Collection**
  [ ] Ticker, timestamp, analysis_type, model_used
  [ ] Full analysis text, summary, key points
  [ ] Confidence scores, sentiment ratings
  [ ] Vector embeddings of analysis content

[ ] **Technical Indicators Collection**
  [ ] Ticker, timestamp, timeframe
  [ ] RSI, MACD, Bollinger Bands values
  [ ] Moving averages, volume indicators
  [ ] Trend strength and direction

[ ] **Support/Resistance Collection**
  [ ] Ticker, timestamp, timeframe
  [ ] Price levels (strong/weak support/resistance)
  [ ] Confidence scores for each level
  [ ] Duration of level validity
  [ ] Historical tests of each level (bounces/breaks)

[ ] **Options Data Collection**
  [ ] Ticker, timestamp, expiration dates
  [ ] Strike prices, call/put prices
  [ ] Greeks, implied volatility
  [ ] Open interest, volume
  [ ] Put-call ratios

[ ] **Prediction Tracking Collection**
  [ ] Prediction ID, ticker, timestamp
  [ ] Predicted price targets and timeframes
  [ ] Actual outcomes (hit/miss)
  [ ] Confidence scores vs. actual accuracy

### 3. Retrieval and Memory Management

#### 3.1 Context-Aware Retrieval System
[ ] **Multi-stage Retrieval Pipeline**
  [ ] Query understanding (identify ticker, timeframe, analysis type)
  [ ] Semantic search for relevant analyses
  [ ] Structured data lookup for specific metrics
  [ ] Recency-based filtering and ranking
  [ ] Relevance scoring and result merging

[ ] **Memory Management**
  [ ] Short-term conversation memory (recent exchanges)
  [ ] Medium-term session memory (current analysis focus)
  [ ] Long-term knowledge base (all historical analyses)
  [ ] Adaptive context window based on conversation flow

[ ] **Cross-Reference System**
  [ ] Link related analyses across time periods
  [ ] Connect technical analysis with options data
  [ ] Map predictions to outcomes
  [ ] Identify contradictions or confirmations between analyses

#### 3.2 Retrieval Augmented Generation (RAG)
[ ] **Enhanced RAG Implementation**
  [ ] Query-specific retrieval from vector and structured databases
  [ ] Dynamic prompt construction with retrieved context
  [ ] Fact-grounding with numerical data
  [ ] Citation of sources and timestamps
  [ ] Confidence scoring based on data recency and relevance

[ ] **Specialized Financial RAG Features**
  [ ] Time-aware retrieval (prioritize recent data for volatile metrics)
  [ ] Trend-aware context (include data showing pattern development)
  [ ] Multi-timeframe analysis (combine insights from different timeframes)
  [ ] Cross-ticker correlation (include relevant data from related stocks)

### 4. LLM Integration with Google Gemini

#### 4.1 Gemini Integration
[ ] **Model Selection and Configuration**
  [ ] Use Gemini 1.5 Pro for complex financial reasoning
  [ ] Fallback to Gemini 1.5 Flash for faster responses
  [ ] Configure system prompts for financial analysis expertise
  [ ] Set appropriate temperature for balanced creativity/accuracy

[ ] **Prompt Engineering**
  [ ] Design specialized prompts for different analysis types
  [ ] Create templates for incorporating retrieved context
  [ ] Develop techniques for numerical reasoning with financial data
  [ ] Implement chain-of-thought prompting for complex analyses

#### 4.2 Response Generation
[ ] **Response Formatting**
  [ ] Structured output for different query types
  [ ] Visualization suggestions based on data type
  [ ] Confidence indicators for different parts of response
  [ ] Clear attribution of sources and timestamps

[ ] **Explanation Generation**
  [ ] Explain reasoning behind recommendations
  [ ] Clarify contradictions between different analyses
  [ ] Provide context for changing predictions over time
  [ ] Generate natural language summaries of technical indicators

### 5. User Interface and Interaction

#### 5.1 Streamlit Chatbot Interface
[ ] **Chat Interface Components**
  [ ] Message history display with formatting for different message types
  [ ] Input area with suggestions and autocomplete
  [ ] Context panel showing active memory elements
  [ ] Quick action buttons for common queries

[ ] **Visualization Integration**
  [ ] Dynamic chart generation based on conversation
  [ ] Interactive elements for exploring data points
  [ ] Side-by-side comparison of historical vs. current analysis
  [ ] Timeline view of changing predictions

#### 5.2 Enhanced User Experience
[ ] **Conversation Management**
  [ ] Save and load conversation sessions
  [ ] Export analysis summaries and insights
  [ ] Bookmark important findings
  [ ] Set alerts for prediction timeframes

[ ] **Personalization Features**
  [ ] User preferences for analysis depth and risk tolerance
  [ ] Watchlist integration for quick access to favorite tickers
  [ ] Customizable dashboard with preferred metrics
  [ ] Learning from user feedback and interactions

### 6. Implementation Plan

#### Phase 1: Data Storage Infrastructure
1. Set up vector database (Chroma) for storing analysis text
2. Extend SQLite database schema for structured financial data
3. Create data ingestion pipeline for all analysis types
4. Implement embedding generation for vector storage
5. Develop data validation and cleaning processes

#### Phase 2: Retrieval System
1. Build query understanding module
2. Implement semantic search against vector database
3. Create structured data lookup functions
4. Develop cross-reference system for related data
5. Build context management system for conversation memory

#### Phase 3: Gemini Integration
1. Set up Google Gemini API integration
2. Design and test specialized financial prompts
3. Implement RAG pipeline with dynamic context retrieval
4. Create response formatting and post-processing
5. Develop fallback mechanisms for API limitations

#### Phase 4: Streamlit Interface
1. Build basic chat interface with message history
2. Implement dynamic visualization generation
3. Create context panel showing active memory
4. Add user preference and personalization features
5. Develop session management and export functionality

#### Phase 5: Testing and Optimization
1. Conduct comprehensive testing with various query types
2. Optimize retrieval performance for large datasets
3. Fine-tune prompts for accuracy and relevance
4. Implement user feedback collection and analysis
5. Develop automated evaluation metrics for response quality

### 7. Technical Considerations

1. **Embedding Models**:
   - Use text-embedding-3-large from OpenAI for high-quality embeddings
   - Consider sentence-transformers for local embedding generation
   - Implement batch processing for efficient embedding updates

2. **Database Scaling**:
   - Start with Chroma (local) for development
   - Plan migration path to Pinecone or Qdrant for production
   - Implement efficient indexing and partitioning strategies

3. **Performance Optimization**:
   - Implement caching for frequent queries
   - Use background workers for embedding generation
   - Consider hybrid search (keyword + semantic) for faster retrieval

4. **Security and Privacy**:
   - Implement proper authentication for API access
   - Ensure secure storage of API keys and credentials
   - Consider data retention policies for user conversations

5. **Integration Points**:
   - Connect with existing analysis pipeline
   - Integrate with data fetching systems
   - Link to visualization components
   - Connect with user feedback mechanisms

### 8. Expected Outcomes

1. **Enhanced User Experience**:
   - Natural conversation about complex financial data
   - Contextually aware responses that reference historical analyses
   - Personalized insights based on user preferences and history

2. **Improved Analysis Quality**:
   - More consistent recommendations grounded in historical data
   - Better explanation of changing market conditions
   - Clearer attribution of sources and confidence levels

3. **Knowledge Accumulation**:
   - Growing knowledge base of market behavior
   - Improving accuracy through continuous learning
   - Developing ticker-specific insights over time

4. **Operational Efficiency**:
   - Faster access to historical analyses
   - Reduced need for redundant analysis
   - Better utilization of existing data assets

### 9. Next Steps

1. Evaluate vector database options (Chroma, Qdrant, Pinecone)
2. Design detailed database schema for all data types
3. Create proof-of-concept for embedding generation and storage
4. Develop initial RAG pipeline with Google Gemini
5. Build basic Streamlit chat interface prototype

## Technical Indicators Enhancement Plan

### 1. Additional Technical Indicators to Implement

We can enhance our technical analysis capabilities by adding the following indicators:

#### 1.1 Trend Indicators
[ ] **Average Directional Index (ADX)**
  [ ] Implement ADX calculation function
  [ ] Add visualization in technical chart
  [ ] Include in LLM analysis prompt

[ ] **Ichimoku Cloud**
  [ ] Calculate Tenkan-sen (Conversion Line)
  [ ] Calculate Kijun-sen (Base Line)
  [ ] Calculate Senkou Span A (Leading Span A)
  [ ] Calculate Senkou Span B (Leading Span B)
  [ ] Calculate Chikou Span (Lagging Span)
  [ ] Add cloud visualization to chart

[ ] **Parabolic SAR**
  [ ] Implement PSAR calculation
  [ ] Add to chart visualization
  [ ] Include in trend analysis

#### 1.2 Momentum Indicators
[ ] **Stochastic Oscillator**
  [ ] Calculate %K and %D lines
  [ ] Add visualization with overbought/oversold zones
  [ ] Include in LLM analysis

[ ] **Commodity Channel Index (CCI)**
  [ ] Implement CCI calculation
  [ ] Add visualization with reference lines
  [ ] Include in momentum analysis

[ ] **Williams %R**
  [ ] Implement Williams %R calculation
  [ ] Add visualization with overbought/oversold zones
  [ ] Include in LLM analysis

#### 1.3 Volume Indicators
[ ] **On-Balance Volume (OBV)**
  [ ] Implement OBV calculation
  [ ] Add visualization to chart
  [ ] Include in volume analysis section

[ ] **Accumulation/Distribution Line**
  [ ] Implement A/D Line calculation
  [ ] Add visualization to chart
  [ ] Include in volume analysis

[ ] **Money Flow Index (MFI)**
  [ ] Implement MFI calculation
  [ ] Add visualization with overbought/oversold zones
  [ ] Include in volume and momentum analysis

#### 1.4 Volatility Indicators
[ ] **Average True Range (ATR)**
  [ ] Implement ATR calculation
  [ ] Add visualization to chart
  [ ] Use for stop-loss recommendations

[ ] **Keltner Channels**
  [ ] Implement Keltner Channels calculation
  [ ] Add visualization to chart
  [ ] Compare with Bollinger Bands for squeeze detection

#### 1.5 Custom/Advanced Indicators
[ ] **VWAP (Volume Weighted Average Price)**
  [ ] Implement VWAP calculation
  [ ] Add visualization to chart
  [ ] Include in institutional analysis section

[ ] **Fibonacci Retracement Levels**
  [ ] Implement automatic Fibonacci level detection
  [ ] Add visualization to chart
  [ ] Include in support/resistance analysis

[ ] **Pivot Points**
  [ ] Implement various pivot point calculations (Standard, Fibonacci, Woodie's, etc.)
  [ ] Add visualization to chart
  [ ] Include in support/resistance analysis

### 2. Implementation Approach

#### 2.1 Core Technical Indicator Module
[ ] Create a dedicated module for technical indicators
  [ ] Design a flexible, extensible architecture
  [ ] Implement base indicator class
  [ ] Create specialized classes for each indicator type
  [ ] Add comprehensive documentation

#### 2.2 Integration with Existing System
[ ] Update the technical chart creation function
  [ ] Add UI controls for new indicators
  [ ] Implement visualization for each indicator
  [ ] Handle subplot management for multiple indicators

[ ] Enhance the LLM analysis function
  [ ] Include new indicators in the technical data dictionary
  [ ] Update the prompt to request analysis of new indicators
  [ ] Add specialized prompts for specific indicator combinations

#### 2.3 Performance Optimization
[ ] Implement caching for indicator calculations
[ ] Use vectorized operations for better performance
[ ] Add progress indicators for long calculations
[ ] Implement lazy loading for rarely used indicators

### 3. User Interface Enhancements

#### 3.1 Indicator Selection and Customization
[ ] Create categorized indicator selection
  [ ] Group indicators by type (Trend, Momentum, Volume, Volatility)
  [ ] Add parameter customization for each indicator
  [ ] Implement presets for common indicator combinations

#### 3.2 Visualization Improvements
[ ] Enhance chart interactivity
  [ ] Add indicator value display on hover
  [ ] Implement crosshair synchronization across subplots
  [ ] Add annotation capabilities for key points

[ ] Create specialized views
  [ ] Multi-timeframe analysis view
  [ ] Indicator correlation view
  [ ] Signal detection and highlighting

### 4. LLM Integration Enhancements

#### 4.1 Indicator-Specific Analysis
[ ] Create specialized prompts for each indicator
[ ] Implement cross-indicator analysis
[ ] Add historical performance context for indicators

#### 4.2 Signal Detection and Interpretation
[ ] Implement automated signal detection
  [ ] Identify crossovers, divergences, and pattern completions
  [ ] Calculate signal strength and reliability
  [ ] Track historical signal performance

[ ] Enhance LLM interpretation
  [ ] Include signal detection results in prompts
  [ ] Request specific analysis of detected signals
  [ ] Ask for confirmation or contradiction of automated signals

### 5. Implementation Priority

1. **First Wave (High Priority)**
   - Stochastic Oscillator
   - ADX
   - OBV
   - ATR
   - Pivot Points

2. **Second Wave (Medium Priority)**
   - Ichimoku Cloud
   - CCI
   - Money Flow Index
   - VWAP
   - Fibonacci Retracement

3. **Third Wave (Lower Priority)**
   - Parabolic SAR
   - Williams %R
   - Accumulation/Distribution Line
   - Keltner Channels

### 6. Technical Considerations

1. **Calculation Accuracy**:
   - Validate all implementations against industry standards
   - Add unit tests for each indicator
   - Compare results with established platforms (TradingView, etc.)

2. **Performance Impact**:
   - Monitor memory usage with multiple indicators
   - Implement progressive loading for heavy calculations
   - Consider using WebAssembly for intensive calculations

3. **Data Requirements**:
   - Ensure sufficient historical data for accurate calculations
   - Handle missing data gracefully
   - Implement proper error handling for edge cases

4. **Visualization Clarity**:
   - Develop a consistent color scheme for indicators
   - Implement proper scaling for each indicator type
   - Ensure readability with multiple indicators active

### 7. Next Steps

1. Create the technical indicators module structure
2. Implement the first wave of high-priority indicators
3. Update the chart visualization function to support new indicators
4. Enhance the LLM prompt to include new indicator analysis
5. Test with various tickers and timeframes