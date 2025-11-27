import datetime
import os
import time
import re
import pytz
from time import mktime
import google.generativeai as genai
import feedparser
import requests
import arxiv

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = 'gemini-2.0-flash-lite' 

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found. Translation will be skipped.")

RSS_SOURCES = {
    "Labs": {
        "Google": "https://blog.google/technology/research/rss/",
        "OpenAI": "https://openai.com/news/rss.xml",
        "Stanford(SAIL)": "https://ai.stanford.edu/blog/feed.xml",
        "UC Berkeley(BAIR)": "https://bair.berkeley.edu/blog/feed.xml",
        "CMU(SCS)": "https://www.cs.cmu.edu/news/feed",
        "MIT(CSAIL)": "https://news.mit.edu/rss/topic/artificial-intelligence2"
    },
    "News": {
        "AI 타임스": "https://cdn.aitimes.com/rss/gn_rss_allArticle.xml",
        "인공지능신문": "https://www.aitimes.kr/rss/allArticle.xml",
        "AI Matters": "https://aimatters.co.kr/feed/"
    }
}

CONFERENCE_LINKS = [
    {"name": "NeurIPS", "url": "https://neurips.cc/", "desc": "Neural Information Processing Systems"},
    {"name": "ICML", "url": "https://icml.cc/", "desc": "International Conference on Machine Learning"},
    {"name": "ICLR", "url": "https://iclr.cc/", "desc": "International Conference on Learning Representations"},
    {"name": "CVPR", "url": "https://cvpr.thecvf.com/", "desc": "Computer Vision and Pattern Recognition"},
    {"name": "AAAI", "url": "https://aaai.org/", "desc": "Assoc. for the Advancement of Artificial Intelligence"}
]

OTHER_LINKS = [
    {"name": "Hugging Face", "url": "https://huggingface.co/", "desc": "The AI Community Building the Future"},
    {"name": "Kaggle", "url": "https://www.kaggle.com/", "desc": "Data Science Competitions"},
    {"name": "Meta", "url": "https://ai.meta.com/research/", "desc": "RSS 제공 안함"},
    {"name": "Anthropic", "url": "https://www.anthropic.com/research", "desc": "RSS 제공 안함"},
    {"name": "xAI", "url": "https://x.ai/news", "desc": "RSS 제공 안함"}
]

SEARCH_QUERY = 'cat:cs.AI OR cat:cs.LG'
MAX_ARXIV_RESULTS = 5
MAX_NEWS_ITEMS = 5
MAX_LABS_ITEMS = 3

KNOWLEDGE_CONTENT = """
<style>
    .timeline-container { padding-left: 10px; margin-top: 20px; text-align: left; }
    .timeline-item { position: relative; padding-left: 30px; padding-bottom: 25px; }
    .timeline-item::before { content: ""; position: absolute; left: 0; top: 0; width: 1px; height: 100%; background-color: #555; }
    .timeline-item:last-child { border-left: none; }
    .timeline-dot { position: absolute; left: -6px; top: 15px; width: 9px; height: 9px; background-color: #1a1a1a; border: 2px solid #999; border-radius: 50%; z-index: 1; }
    .knowledge-paper details { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; margin-top: 0; padding-top: 0; border-top: 1px solid #333; }
    .knowledge-paper summary { padding: 10px 15px; font-weight: bold; cursor: pointer; list-style: none; display: flex; justify-content: space-between; align-items: center; color: #ddd; font-size: 1em; line-height: 1.2; min-height: 20px; }
    .knowledge-paper summary::-webkit-details-marker { display: none; }
    .knowledge-paper summary::after { content: '+'; font-size: 1.2em; color: #777; margin-left: 10px; display: flex; align-items: center; height: 100%; line-height: 1; }
    .knowledge-paper details[open] summary::after { content: '-'; }
    .timeline-content { padding: 10px 15px 20px 15px; border-top: 1px solid #2a2a2a; color: #bbb; font-size: 0.9em; line-height: 1.6; }
    .event-list { margin: 0; padding-left: 18px; }
    .event-list li { margin-bottom: 6px; }
    .timeline-content strong { color: #fff; font-weight: 700; }
</style>

<h3>AI History Timeline</h3>
<div class="timeline-container">
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>1950: AI의 태동</summary><div class="timeline-content"><ul class="event-list"><li><strong>앨런 튜링</strong>, "계산 기계와 지능" 논문 발표 및 튜링 테스트 제안.</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>1956: 용어의 탄생</summary><div class="timeline-content"><ul class="event-list"><li><strong>다트머스 회의</strong> 개최, John McCarthy가 'Artificial Intelligence' 용어 최초 사용.</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>2012: 딥러닝 혁명</summary><div class="timeline-content"><ul class="event-list"><li><strong>AlexNet</strong> 이미지넷 우승, CNN 성능 입증.</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>2016: 알파고 모멘트</summary><div class="timeline-content"><ul class="event-list"><li>구글 딥마인드 <strong>AlphaGo</strong> vs 이세돌.</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>2017: Transformer</summary><div class="timeline-content"><ul class="event-list"><li>구글 "Attention Is All You Need" 논문 발표 (LLM의 시초).</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>2022: 생성형 AI (GenAI)</summary><div class="timeline-content"><ul class="event-list"><li>OpenAI <strong>ChatGPT</strong> 공개, Stable Diffusion 등장.</li></ul></div></details></div>
    <div class="timeline-item"><div class="timeline-dot"></div><details><summary>2024: 멀티모달의 시대</summary><div class="timeline-content"><ul class="event-list"><li><strong>GPT-4o, Gemini 1.5</strong> 등 실시간 멀티모달 모델 등장.</li></ul></div></details></div>
</div>
"""

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.replace("&nbsp;", " ").strip()

def get_feed_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return feedparser.parse(response.content)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def collect_rss_data():
    data = {}
    print("RSS 피드 수집 중...")
    
    for category, feeds in RSS_SOURCES.items():
        items = []
        if category == "News":
            limit = MAX_NEWS_ITEMS
        else:
            limit = MAX_LABS_ITEMS
            
        for name, url in feeds.items():
            feed = get_feed_data(url)
            if feed and feed.entries:
                for entry in feed.entries[:limit]:
                    published_parsed = None
                    date_str = ""
                    
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_parsed = entry.published_parsed
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_parsed = entry.updated_parsed
                    
                    if published_parsed:
                        dt = datetime.datetime.fromtimestamp(mktime(published_parsed))
                        date_str = dt.strftime("%Y-%m-%d")
                    else:
                        published_parsed = time.localtime(0)
                        date_str = "Recent"

                    summary = clean_html(getattr(entry, 'summary', entry.title))
                    
                    items.append({
                        "source": name,
                        "title": entry.title,
                        "link": entry.link,
                        "date": date_str,
                        "timestamp": published_parsed,
                        "summary": summary
                    })
            else:
                print(f"Failed to fetch: {name}")
        
        items.sort(key=lambda x: x['timestamp'], reverse=True)
        data[category] = items
        
    return data

def get_arxiv_papers():
    print(f"ArXiv 논문 검색 중...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_ARXIV_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    kst = pytz.timezone('Asia/Seoul')
    
    try:
        for r in client.results(search):
            try:
                pub_date = r.published.astimezone(kst).strftime("%Y-%m-%d")
            except:
                pub_date = r.published.strftime("%Y-%m-%d")
            
            results.append({
                "title": r.title.replace('\n', ' '),
                "url": r.entry_id,
                "date": pub_date,
                "category": r.primary_category,
                "authors": r.authors[0].name + (" et al." if len(r.authors)>1 else ""),
                "abstract": r.summary.replace("\n", " ")
            })
    except Exception as e:
        print(f"ArXiv Error: {e}")

    return results

def translate_with_gemini(text):
    if not GOOGLE_API_KEY or not text:
        return text

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
        Translate the following text into Korean perfectly.
        
        Rules:
        1. Keep technical AI terms in English (e.g., LLM, Transformer, Diffusion, RAG, Zero-shot).
        2. Make the Korean sentence sound natural and professional.
        3. Do NOT provide explanations, just the translated text.
        4. If the text is a title, keep it concise.
        
        Text to translate:
        {text}
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def process_translation(data_list, fields):
    if not GOOGLE_API_KEY:
        return

    print(f"Translating {len(data_list)} items... (Please wait)")
    for i, item in enumerate(data_list):
        for field in fields:
            if field in item:
                original = item[field]
                translated = translate_with_gemini(original)
                item[field] = translated
                time.sleep(4)

def create_html(rss_data, paper_data, conf_links, other_links, knowledge_content):
    now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    
    content_html = ""
    
    for tab_key, tab_id in [("News", "News"), ("Labs", "Labs")]:
        if tab_key in rss_data:
            cards = ""
            for item in rss_data[tab_key]:
                cards += f'''
                <div class="card">
                    <div class="card-meta">
                        <span class="badge">{item['source']}</span>
                        <span class="date">{item['date']}</span>
                    </div>
                    <a href="{item['link']}" target="_blank" class="card-title">{item['title']}</a>
                </div>
                '''
            content_html += f'<div id="{tab_id}" class="tab-content"><div class="card-grid">{cards}</div></div>'

    paper_cards = ""
    if paper_data:
        for paper in paper_data:
            paper_cards += f'''
            <div class="card paper-card">
                <div class="card-meta">
                    <span class="badge paper-badge">{paper['category']}</span>
                    <span class="date">{paper['date']}</span>
                </div>
                <a href="{paper['url']}" target="_blank" class="card-title">{paper['title']}</a>
                <div class="card-author">{paper['authors']}</div>
                <details>
                    <summary>요약 (Abstract)</summary>
                    <div class="abstract-text">{paper['abstract']}</div>
                </details>
            </div>
            '''
    
    ack_msg = '<div class="arxiv-ack">"Thank you to arXiv for use of its open access interoperability."</div>'
    content_html += f'<div id="Papers" class="tab-content"><div class="card-grid">{paper_cards}</div>{ack_msg}</div>'

    content_html += f'''
    <div id="Knowledge" class="tab-content">
        <div class="knowledge-paper">
            {knowledge_content}
        </div>
    </div>
    '''

    conf_cards = ""
    for item in conf_links:
        conf_cards += f'''
        <a href="{item['url']}" target="_blank" class="card link-card">
            <div class="link-title">{item['name']} ↗</div>
            <div class="link-desc">{item['desc']}</div>
        </a>
        '''
    content_html += f'<div id="Conferences" class="tab-content"><div class="card-grid">{conf_cards}</div></div>'

    link_cards = ""
    for item in other_links:
        link_cards += f'''
        <a href="{item['url']}" target="_blank" class="card link-card">
            <div class="link-title">{item['name']} ↗</div>
            <div class="link-desc">{item['desc']}</div>
        </a>
        '''
    content_html += f'<div id="Others" class="tab-content"><div class="card-grid">{link_cards}</div></div>'

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily AI Insights</title>
        <style>
            :root {{
                --bg: #111111;
                --card-bg: #1e1e1e;
                --text-main: #e0e0e0;
                --text-sub: #9e9e9e;
                --accent: #4dabf7; 
                --border: #333;
            }}
            body {{
                background-color: var(--bg);
                color: var(--text-main);
                font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0; padding: 20px;
                line-height: 1.6;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; min-height: 100vh; display: flex; flex-direction: column; }}
            
            header {{ text-align: center; margin-bottom: 30px; border-bottom: 1px solid #222; padding-bottom: 20px; }}
            h1 {{ margin: 0; font-weight: 800; font-size: 2.2rem; letter-spacing: -1px; background: linear-gradient(to right, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .timestamp {{ color: #555; font-size: 0.85rem; font-family: monospace; margin-top: 8px; }}
            
            .tabs {{ 
                display: flex; gap: 10px; margin-bottom: 25px; 
                padding-bottom: 10px; overflow-x: auto; white-space: nowrap; justify-content: center;
            }}
            .tabs::-webkit-scrollbar {{ display: none; }}

            .tab-btn {{
                background: transparent; border: 1px solid #444; color: var(--text-sub);
                padding: 8px 16px; cursor: pointer; border-radius: 20px; font-weight: 600; font-size: 0.9rem;
                transition: all 0.2s; flex: 0 0 auto;
            }}
            .tab-btn:hover {{ border-color: var(--accent); color: #fff; }}
            .tab-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
            
            .tab-content {{ display: none; animation: fadeIn 0.4s; flex: 1; }}
            .tab-content.active {{ display: block; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            
            .card-grid {{
                display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
                gap: 20px; justify-content: center; 
            }}
            
            .card {{
                background-color: var(--card-bg); border: 1px solid var(--border);
                border-radius: 12px; padding: 24px; transition: transform 0.2s, box-shadow 0.2s;
                display: flex; flex-direction: column;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .card:hover {{ transform: translateY(-3px); border-color: #555; box-shadow: 0 8px 12px rgba(0,0,0,0.2); }}
            .card-meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 0.8rem; }}
            
            .badge {{ background: #2a2a2a; color: #ccc; padding: 4px 10px; border-radius: 6px; font-weight: 600; border: 1px solid #333; }}
            .paper-badge {{ color: var(--accent); border-color: #2a3a4a; background: #1a202a; }} 
            
            .date {{ color: #666; font-family: monospace; }}
            
            .card-title {{ 
                display: block; font-size: 1.15rem; font-weight: 700; color: #fff; 
                text-decoration: none; margin-bottom: 8px; 
                line-height: 1.4; 
                word-break: keep-all; 
                word-wrap: break-word;
                flex-grow: 1; 
            }}
            .card-title:hover {{ color: var(--accent); text-decoration: underline; }}
            .card-author {{ font-size: 0.9rem; color: #777; margin-bottom: 12px; font-style: italic; }}
            
            .link-card {{ text-align: center; display: flex; flex-direction: column; justify-content: center; min-height: 140px; text-decoration: none; align-items: center; }}
            .link-title {{ font-size: 1.4rem; font-weight: bold; margin-bottom: 6px; color: #fff; }}
            .link-desc {{ font-size: 0.9rem; color: var(--text-sub); }}
            
            .knowledge-paper {{ background-color: #1a1a1a; border: 1px solid var(--border); border-radius: 12px; padding: 30px; min-height: 400px; color: #ddd; }}
            .knowledge-paper h3 {{ border-bottom: 2px solid var(--accent); padding-bottom: 15px; margin-top: 0; }}
            
            details {{ margin-top: 15px; border-top: 1px solid #333; padding-top: 15px; }}
            summary {{ cursor: pointer; color: #888; font-size: 0.9rem; font-weight: 600; transition: color 0.2s; }}
            summary:hover {{ color: var(--accent); }}
            .abstract-text {{ margin-top: 12px; font-size: 0.95rem; color: #bbb; text-align: justify; line-height: 1.65; word-break: break-all; }}
            
            .arxiv-ack {{ text-align: center; margin-top: 40px; margin-bottom: 20px; color: #444; font-size: 0.8rem; font-family: monospace; padding-top: 20px; border-top: 1px solid #222; }}
            
            @media (max-width: 480px) {{
                body {{ padding: 15px; }}
                h1 {{ font-size: 1.8rem; }}
                .card-grid {{ grid-template-columns: 1fr; }} 
                .tabs {{ justify-content: flex-start; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Daily AI Insights</h1>
                <div class="timestamp">Last Updated: {now_kst}</div>
            </header>

            <nav class="tabs">
                <button class="tab-btn active" onclick="openTab('News')">뉴스 (News)</button>
                <button class="tab-btn" onclick="openTab('Labs')">연구소 (Labs)</button>
                <button class="tab-btn" onclick="openTab('Papers')">논문 (Papers)</button>
                <button class="tab-btn" onclick="openTab('Conferences')">학회 (Conf)</button>
                <button class="tab-btn" onclick="openTab('Knowledge')">지식 (History)</button>
                <button class="tab-btn" onclick="openTab('Others')">기타 (Etc)</button>
            </nav>

            <main>
                {content_html}
            </main>
        </div>

        <script>
            function openTab(tabName) {{
                var i;
                var x = document.getElementsByClassName("tab-content");
                for (i = 0; i < x.length; i++) {{
                    x[i].classList.remove("active");
                }}
                document.getElementById(tabName).classList.add("active");
                
                var btns = document.getElementsByClassName("tab-btn");
                for (i = 0; i < btns.length; i++) {{
                    btns[i].classList.remove("active");
                    if (btns[i].getAttribute('onclick').includes(tabName)) {{
                         btns[i].classList.add("active");
                    }}
                }}
            }}
            openTab('News'); 
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    rss_data = collect_rss_data()
    paper_data = get_arxiv_papers()

    if GOOGLE_API_KEY:
        print("\n--- 번역 프로세스 시작 (약간의 시간이 소요됩니다) ---")
        
        if 'Labs' in rss_data:
            print("1. 해외 연구소 블로그 제목 번역 중...")
            process_translation(rss_data['Labs'], ['title'])

        if paper_data:
            print("2. ArXiv 논문 제목 및 요약 번역 중...")
            process_translation(paper_data, ['title', 'abstract'])
            
        print("--- 번역 완료 ---\n")
    else:
        print("\n[알림] GOOGLE_API_KEY가 설정되지 않아 번역이 생략됩니다.\n")

    html_out = create_html(rss_data, paper_data, CONFERENCE_LINKS, OTHER_LINKS, KNOWLEDGE_CONTENT)
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)
        
    print("index.html 파일이 성공적으로 생성되었습니다.")
