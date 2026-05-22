from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
import pandas as pd
import json
import os
import pickle
from datetime import datetime, timedelta

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
SITE_URL = 'https://www.sukaraliving.com/'

# ── GSC Authentication ─────────────────────────────────────────────────────────

def get_gsc_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as f:
            pickle.dump(creds, f)
    return build('searchconsole', 'v1', credentials=creds)

service = get_gsc_service()

# ── GSC Tools ──────────────────────────────────────────────────────────────────

@tool
def get_top_queries(days: int = 28) -> str | list:
    """Get the top search queries driving traffic to the blog.
    Returns clicks, impressions, CTR, and average position for each query.
    days = number of past days to analyse (default 28)."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query'],
            'rowLimit': 20,
            'orderBy': [{'fieldName': 'clicks', 'sortOrder': 'DESCENDING'}]
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No data yet — GSC takes a few days to populate after site launch."
    df = pd.DataFrame([{
        'query': r['keys'][0],
        'clicks': r['clicks'],
        'impressions': r['impressions'],
        'ctr': f"{r['ctr']*100:.1f}%",
        'position': f"{r['position']:.1f}"
    } for r in rows])
    return df.to_string(index=False)

@tool
def get_top_pages(days: int = 28) -> str | list:
    """Get the top performing pages on the blog by clicks.
    days = number of past days to analyse (default 28)."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['page'],
            'rowLimit': 10,
            'orderBy': [{'fieldName': 'clicks', 'sortOrder': 'DESCENDING'}]
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No page data yet — check back in a few days."
    df = pd.DataFrame([{
        'page': r['keys'][0].replace(SITE_URL, '/'),
        'clicks': r['clicks'],
        'impressions': r['impressions'],
        'ctr': f"{r['ctr']*100:.1f}%",
        'position': f"{r['position']:.1f}"
    } for r in rows])
    return df.to_string(index=False)

@tool
def get_overall_performance(days: int = 28) -> str | list:
    """Get the overall site performance summary — total clicks, impressions,
    average CTR, and average position for the given period.
    days = number of past days (default 28)."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['date'],
            'rowLimit': 90,
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No performance data yet."
    total_clicks = sum(r['clicks'] for r in rows)
    total_impressions = sum(r['impressions'] for r in rows)
    avg_ctr = sum(r['ctr'] for r in rows) / len(rows) * 100
    avg_pos = sum(r['position'] for r in rows) / len(rows)
    return (
        f"Period: {start_date} to {end_date}\n"
        f"Total clicks:       {total_clicks}\n"
        f"Total impressions:  {total_impressions}\n"
        f"Average CTR:        {avg_ctr:.2f}%\n"
        f"Average position:   {avg_pos:.1f}"
    )

@tool
def get_keyword_opportunity(days: int = 28) ->  str | list:
    """Find keyword opportunities — queries with high impressions but low clicks
    (high position number = ranking low = opportunity to improve content).
    days = number of past days (default 28)."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query'],
            'rowLimit': 100,
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No data yet."
    opportunities = [r for r in rows if r['impressions'] > 10 and r['position'] > 10]
    opportunities.sort(key=lambda x: x['impressions'], reverse=True)
    if not opportunities:
        return "No clear opportunities found yet — need more impressions data."
    df = pd.DataFrame([{
        'query': r['keys'][0],
        'impressions': r['impressions'],
        'clicks': r['clicks'],
        'position': f"{r['position']:.0f}",
        'opportunity': 'High' if r['position'] > 20 else 'Medium'
    } for r in opportunities[:15]])
    return df.to_string(index=False)

tools = [get_top_queries, get_top_pages, get_overall_performance, get_keyword_opportunity]
tools_map = {t.name: t for t in tools}

# ── LangChain Agent ────────────────────────────────────────────────────────────

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(history: list[BaseMessage]) -> str | list:
    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)
        if not response.tool_calls:
            return response.content
        for tool_call in response.tool_calls:
            print(f"  [fetching: {tool_call['name']}...]")
            result = tools_map[tool_call['name']].invoke(tool_call['args'])
            history.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

# ── Conversation Loop ──────────────────────────────────────────────────────────

chat_history: list[BaseMessage] = [
    SystemMessage(content="""You are an expert SEO and content strategist with access to 
    Google Search Console data for the blog 'Simply' at simply-astha.netlify.app — 
    a blog about minimalism, lifestyle, and sustainable living.
    
    When asked about performance, always fetch the relevant GSC data first, then provide 
    specific, actionable insights. Point out what's working, what needs improvement, 
    and suggest concrete next steps for the blog's growth.""")
]

print("GSC Agent ready. Try asking:")
print("  - How is my blog performing?")
print("  - What are my top keywords?")
print("  - Which pages get the most traffic?")
print("  - Where are my keyword opportunities?")
print("  - Give me a full SEO analysis\n")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        break
    chat_history.append(HumanMessage(content=user_input))
    answer = run_agent(chat_history)
    chat_history.append(AIMessage(content=answer))
    print(f"\nAgent: {answer}\n")