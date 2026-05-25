# Sukara Living — SEO AI Agent

An AI-powered SEO analysis agent built with LangChain and LangGraph for 
[Sukara Living](https://www.sukaraliving.in/) — a minimalism and sustainable 
living blog.

## What it does
- Pulls live data from Google Search Console (clicks, impressions, rankings)
- Searches the web for trending content and competitor analysis
- Identifies keyword opportunities and content gaps
- Suggests data-driven blog post ideas

## Agents

### search_gsc_agent.py
Full-featured agent using **OpenAI GPT-4o-mini** + **Tavily web search**.  
Tools: top queries, top pages, keyword opportunities, site performance, 
web search, content ideas, keyword research.

### gsc_agent.py
Lightweight agent using **Groq LLaMA 3 70B** — GSC data analysis only.  
Tools: top queries, top pages, keyword opportunities, site performance, 
queries by page.

## Tech Stack
- Python, LangChain, LangGraph
- OpenAI GPT-4o-mini / Groq LLaMA 3
- Google Search Console API
- Tavily Search API

## Setup
1. Clone the repo
2. Add your credentials: `credentials.json` (GSC OAuth), `.env` with API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python search_gsc_agent.py`

## Built by
Astha Shrimali — SEO & AI Specialist at Dentsu  
[LinkedIn](https://linkedin.com/in/your-profile) · [Blog](https://sukaraliving.in)
