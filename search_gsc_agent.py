from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
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

# ── Tavily Web Search Tool ─────────────────────────────────────────────────────

from tavily import TavilyClient
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query: str) -> list:
    """Helper that always returns a clean list of results."""
    try:
        response = tavily_client.search(query=query, max_results=5)
        return response.get("results", [])
    except Exception as e:
        return [{"title": "Error", "url": "", "content": str(e)}]

@tool
def search_web(query: str) -> str:
    """Search the web for current information, trends, news, or competitor
    content. Use this to research keywords, find content ideas, check what
    competitors are writing about, or get context on any topic."""
    results = tavily_search(query)
    if not results:
        return "No results found."
    output = []
    for i, r in enumerate(results[:5], 1):
        output.append(
            f"{i}. {r.get('title', 'No title')}\n"
            f"   URL: {r.get('url', '')}\n"
            f"   {r.get('content', '')[:300]}\n"
        )
    return "\n".join(output)

@tool
def search_content_ideas(topic: str) -> str:
    """Search for content ideas, trending articles, and popular posts
    on a given topic related to minimalism or sustainable living."""
    queries = [
        f"{topic} minimalism sustainable living 2025",
        f"best {topic} tips blog",
        f"{topic} trending articles"
    ]
    all_results = []
    for q in queries:
        results = tavily_search(q)
        for r in results[:2]:
            all_results.append(
                f"• {r.get('title', '')}\n"
                f"  {r.get('url', '')}"
            )
    return "\n".join(all_results[:10]) if all_results else "No ideas found."

@tool
def research_keyword(keyword: str) -> str:
    """Research a specific keyword — find what content already exists,
    what angle competitors take, and what questions people are asking."""
    results = tavily_search(f"{keyword} blog article guide 2025")
    if not results:
        return "No results found."
    output = [f"Top content for '{keyword}':\n"]
    for i, r in enumerate(results[:5], 1):
        output.append(
            f"{i}. {r.get('title', '')}\n"
            f"   {r.get('url', '')}\n"
            f"   Summary: {r.get('content', '')[:200]}\n"
        )
    return "\n".join(output)

# ── All Tools Together ─────────────────────────────────────────────────────────

tools = [
    get_top_queries,
    get_top_pages,
    get_keyword_opportunities,
    get_overall_performance,
    search_web,
    search_content_ideas,
    research_keyword,
]
tools_map = {t.name: t for t in tools}

# ── LangChain Agent ────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(history: list[BaseMessage]) -> str | list:
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
    SystemMessage(content="""You are an expert SEO strategist and content advisor 
    for Sukara Living (https://www.sukaraliving.com/) — a blog about minimalism, 
    intentional lifestyle, and sustainable living.

    You have two types of tools available:
    
    1. GSC tools — fetch real performance data from Google Search Console 
       (clicks, impressions, keyword rankings, page performance)
    
    2. Web search tools — search the internet for trends, competitor content, 
       keyword research, and content ideas
    
    When answering questions:
    - Always fetch real data first before giving advice
    - Combine GSC data with web research for deeper insights
    - Give specific, actionable recommendations for the blog
    - Suggest concrete content ideas based on what you find
    - Be concise but thorough""")
]

print("=" * 55)
print("  Sukara Living — SEO + Web Search Agent")
print("=" * 55)
print("\nThis agent can:")
print("  • Pull your GSC data (clicks, keywords, rankings)")
print("  • Search the web for trends and content ideas")
print("  • Research keywords before you write a post")
print("  • Find what competitors are writing about")
print("  • Suggest new blog post ideas based on real data\n")
print("Try asking:")
print("  - What should I write about next?")
print("  - Research the keyword 'capsule wardrobe 2025'")
print("  - How is my blog doing and what topics are trending?")
print("  - Find content ideas about sustainable living")
print("  - Compare my GSC keywords with what's trending\n")
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