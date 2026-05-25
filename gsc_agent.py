
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
 
load_dotenv()
 
SCOPES   = ['https://www.googleapis.com/auth/webmasters.readonly']
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
def get_top_queries(days: int = 28) -> str:
    """Get top search queries driving traffic to Sukara Living blog.
    Returns clicks, impressions, CTR and position for each query."""
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start, 'endDate': end,
            'dimensions': ['query'], 'rowLimit': 20,
            'orderBy': [{'fieldName': 'clicks', 'sortOrder': 'DESCENDING'}]
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No GSC data yet — site may still be getting indexed."
    df = pd.DataFrame([{
        'query':       r['keys'][0],
        'clicks':      r['clicks'],
        'impressions': r['impressions'],
        'ctr':         f"{r['ctr']*100:.1f}%",
        'position':    f"{r['position']:.1f}"
    } for r in rows])
    return df.to_string(index=False)
 
@tool
def get_top_pages(days: int = 28) -> str:
    """Get top performing pages on Sukara Living by clicks."""
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start, 'endDate': end,
            'dimensions': ['page'], 'rowLimit': 10,
            'orderBy': [{'fieldName': 'clicks', 'sortOrder': 'DESCENDING'}]
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No page data yet."
    df = pd.DataFrame([{
        'page':        r['keys'][0].replace(SITE_URL, '/'),
        'clicks':      r['clicks'],
        'impressions': r['impressions'],
        'ctr':         f"{r['ctr']*100:.1f}%",
        'position':    f"{r['position']:.1f}"
    } for r in rows])
    return df.to_string(index=False)
 
@tool
def get_keyword_opportunities(days: int = 28) -> str:
    """Find keywords with high impressions but low clicks —
    these are SEO opportunities where content can be improved."""
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start, 'endDate': end,
            'dimensions': ['query'], 'rowLimit': 100,
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No data yet."
    opps = [r for r in rows if r['impressions'] > 5 and r['position'] > 10]
    opps.sort(key=lambda x: x['impressions'], reverse=True)
    if not opps:
        return "No clear keyword opportunities found yet — need more data."
    df = pd.DataFrame([{
        'query':       r['keys'][0],
        'impressions': r['impressions'],
        'clicks':      r['clicks'],
        'position':    f"{r['position']:.0f}",
        'opportunity': 'High' if r['position'] > 20 else 'Medium'
    } for r in opps[:15]])
    return df.to_string(index=False)
 
@tool
def get_overall_performance(days: int = 28) -> str:
    """Get overall site performance — total clicks, impressions,
    average CTR and average position for Sukara Living."""
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start, 'endDate': end,
            'dimensions': ['date'], 'rowLimit': 90,
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return "No performance data yet."
    total_clicks      = sum(r['clicks'] for r in rows)
    total_impressions = sum(r['impressions'] for r in rows)
    avg_ctr           = sum(r['ctr'] for r in rows) / len(rows) * 100
    avg_pos           = sum(r['position'] for r in rows) / len(rows)
    return (
        f"Period:             {start} to {end}\n"
        f"Total clicks:       {total_clicks}\n"
        f"Total impressions:  {total_impressions}\n"
        f"Average CTR:        {avg_ctr:.2f}%\n"
        f"Average position:   {avg_pos:.1f}"
    )
 
@tool
def get_queries_by_page(page_path: str, days: int = 28) -> str:
    """Get all search queries that lead to a specific page.
    Pass a page path like '/articles/capsule-wardrobe.html'"""
    end   = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    full_url = SITE_URL.rstrip('/') + page_path
    response = service.searchanalytics().query(
        siteUrl=SITE_URL,
        body={
            'startDate': start, 'endDate': end,
            'dimensions': ['query'],
            'rowLimit': 20,
            'dimensionFilterGroups': [{
                'filters': [{
                    'dimension': 'page',
                    'operator': 'equals',
                    'expression': full_url
                }]
            }],
            'orderBy': [{'fieldName': 'impressions', 'sortOrder': 'DESCENDING'}]
        }
    ).execute()
    rows = response.get('rows', [])
    if not rows:
        return f"No query data found for {page_path}"
    df = pd.DataFrame([{
        'query':       r['keys'][0],
        'clicks':      r['clicks'],
        'impressions': r['impressions'],
        'position':    f"{r['position']:.1f}"
    } for r in rows])
    return f"Queries for {page_path}:\n" + df.to_string(index=False)
 
# ── All Tools Together ─────────────────────────────────────────────────────────
 
tools = [
    get_top_queries,
    get_top_pages,
    get_keyword_opportunities,
    get_overall_performance,
    get_queries_by_page,
]
tools_map = {t.name: t for t in tools}
 
# ── Groq LLaMA Agent ──────────────────────────────────────────────────────────
 
from pydantic import SecretStr

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or "")
)
llm_with_tools = llm.bind_tools(tools)
 
def run_agent(history: list[BaseMessage]) -> str:
    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)
 
        tool_calls = getattr(response, 'tool_calls', None)
        if not tool_calls:
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    return ' '.join(
                        block.get('text', '') if isinstance(block, dict)
                        else str(block) for block in content
                    )
                return str(content)
            return str(response)
 
        for tool_call in tool_calls:
            print(f"  [using: {tool_call['name']} — {list(tool_call['args'].values())}]")
            result = tools_map[tool_call['name']].invoke(tool_call['args'])
            history.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            ))
 
# ── Conversation Loop ──────────────────────────────────────────────────────────
 
chat_history: list[BaseMessage] = [
    SystemMessage(content="""You are an SEO analyst for Sukara Living 
    (https://www.sukaraliving.com/) — a blog about minimalism, intentional 
    lifestyle, and sustainable living.
 
    You have access to Google Search Console data tools. Use them to:
    - Fetch real performance data before giving any advice
    - Identify which pages are performing well and which need improvement
    - Find keyword opportunities the blog is missing
    - Give specific, actionable recommendations based on actual data
    - Suggest improvements to existing articles based on their query data
 
    Always be concise, data-driven, and specific to Sukara Living's niche.""")
]
 
print("=" * 55)
print("  Sukara Living — GSC Agent (Groq LLaMA)")
print("=" * 55)
print("\nThis agent can:")
print("  • Pull your GSC data (clicks, keywords, rankings)")
print("  • Find keyword opportunities hiding in your data")
print("  • Show which queries lead to each specific page")
print("  • Give data-driven content improvement advice\n")
print("Try asking:")
print("  - How is my blog performing this month?")
print("  - Which pages need the most improvement?")
print("  - What keywords am I ranking for but not clicking?")
print("  - Show me queries for /articles/capsule-wardrobe.html")
print("  - Which article should I update first?\n")
print("Type 'quit' to exit.\n")
 
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        break
    if not user_input:
        continue
    chat_history.append(HumanMessage(content=user_input))
    print()
    answer = run_agent(chat_history)
    chat_history.append(AIMessage(content=str(answer)))
    print(f"\nAgent: {answer}\n")
    print("-" * 55 + "\n")