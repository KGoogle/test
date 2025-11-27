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
from email.utils import parsedate_to_datetime

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = 'gemini-2.0-flash-lite' 

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

RSS_SOURCES = {
    "Labs": {
        "Google": "https://blog.google/technology/research/rss/",
        "OpenAI": "https://openai.com/news/rss.xml",
        "UC Berkeley(BAIR)": "https://bair.berkeley.edu/blog/feed.xml",
        "CMU(SCS)": "https://www.cs.cmu.edu/news/feed",
        "MIT(CSAIL)": "https://news.mit.edu/rss/topic/artificial-intelligence2"
    },
    "News": {
        "AI 타임스": "https://cdn.aitimes.com/rss/gn_rss_allArticle.xml",
        "인공지능신문": "https://www.aitimes.kr/rss/S1N2.xml",
        "AI Matters": "https://aimatters.co.kr/category/news-report/ai-report/feed/"
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
    {"name": "xAI", "url": "https://x.ai/news", "desc": "RSS 제공 안함"},
    {"name": "Stanford(SAIL)", "url": "https://ai.stanford.edu/blog/", "desc": "RSS 제공 안함"}
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

def parse_custom_date(entry):
    """
    다양한 포맷의 날짜를 파싱하고, 최종적으로 한국 시간(KST) datetime 객체를 반환합니다.
    """
    date_str = getattr(entry, 'published', "") or getattr(entry, 'updated', "") or getattr(entry, 'date', "")
    dt_obj = None
    kst = pytz.timezone('Asia/Seoul')

    if not dt_obj and date_str:
        try:
            dt_obj = parsedate_to_datetime(date_str)
        except:
            pass

    if not dt_obj and date_str:
        try:
            dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            dt_obj = kst.localize(dt_obj) 
        except ValueError:
            pass

    if not dt_obj:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            dt_obj = datetime.datetime.fromtimestamp(mktime(entry.published_parsed))
            if not dt_obj.tzinfo:
                dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)

    if not dt_obj:
        dt_obj = datetime.datetime.now(kst)

    if dt_obj.tzinfo:
        return dt_obj.astimezone(kst)
    else:
        return kst.localize(dt_obj)

def collect_rss_data():
    data = {}
    print("RSS 피드 수집 및 시간 변환 중...")
    
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
                    
                    dt_obj = parse_custom_date(entry)
                    
                    if category == "Labs":
                        date_str = dt_obj.strftime("%Y-%m-%d")
                    else:
                        date_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                    
                    try:
                        timestamp = dt_obj.timestamp()
                    except:
                        timestamp = time.time()

                    summary = clean_html(getattr(entry, 'summary', entry.title))
                    
                    items.append({
                        "source": name,
                        "title": entry.title,
                        "link": entry.link,
                        "date": date_str,
                        "timestamp": timestamp,
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
        prompt = f"Translate the following text into Korean naturally, keeping technical AI terms (like LLM, Transformer) intact:\n\n{text}"
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def process_translation(data_list, fields):
    if not GOOGLE_API_KEY:
        return

    print(f"Translating {len(data_list)} items... (It may take a while)")
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
                    <summary>Abstract (Translated)</summary>
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
        <title>AI Information</title>
        <style>
            :root {{
                --bg: #111111;
                --card-bg: #1c1c1c;
                --text-main: #f0f0f0;
                --text-sub: #888888;
                --accent: #ffffff; 
                --border: #333;
            }}
            body {{
                background-color: var(--bg);
                color: var(--text-main);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0; padding: 20px;
                line-height: 1.6;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; min-height: 100vh; display: flex; flex-direction: column; }}
            
            header {{ text-align: center; margin-bottom: 20px; }}
            h1 {{ margin: 0; font-weight: 700; font-size: 2rem; }}
            .timestamp {{ color: var(--text-sub); font-size: 0.8rem; font-family: monospace; margin-top: 10px; }}
            
            .tabs {{ 
                display: flex; gap: 12px; margin-bottom: 30px; border-bottom: 1px solid var(--border);
                padding-bottom: 15px; overflow-x: auto; white-space: nowrap; justify-content: center;
                -webkit-overflow-scrolling: touch; scrollbar-width: none;
            }}
            .tabs::-webkit-scrollbar {{ display: none; }}

            .tab-btn {{
                background: transparent; border: 1px solid var(--border); color: var(--text-sub);
                padding: 10px 20px; cursor: pointer; border-radius: 8px; font-weight: 600; font-size: 0.95rem;
                transition: all 0.2s; flex: 0 0 auto; 
            }}
            .tab-btn:hover {{ border-color: var(--accent); color: #fff; }}
            .tab-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
            
            .tab-content {{ display: none; animation: fadeIn 0.3s; flex: 1; }}
            .tab-content.active {{ display: block; }}
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            
            .card-grid {{
                display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); 
                gap: 20px; justify-content: center; 
            }}
            
            .card {{
                background-color: var(--card-bg); border: 1px solid var(--border);
                border-radius: 10px; padding: 20px; transition: transform 0.2s;
                display: flex; flex-direction: column;
            }}
            .card:hover {{ transform: translateY(-3px); border-color: #555; }}
            .card-meta {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.75rem; }}
            
            .badge {{ background: #222; color: var(--accent); padding: 3px 8px; border-radius: 4px; font-weight: bold; border: 1px solid #333; }}
            .paper-badge {{ color: var(--accent); }} 
            
            .date {{ color: #666; font-family: monospace; }}
            .card-title {{ display: block; font-size: 1.1rem; font-weight: 600; color: #fff; text-decoration: none; margin-bottom: 5px; word-break: keep-all; flex-grow: 1; }}
            .card-title:hover {{ color: var(--accent); text-decoration: underline; }}
            .card-author {{ font-size: 0.85rem; color: #777; margin-bottom: 8px; font-style: italic; }}
            
            .link-card {{ text-align: center; display: flex; flex-direction: column; justify-content: center; min-height: 120px; text-decoration: none; align-items: center; }}
            .link-title {{ font-size: 1.3rem; font-weight: bold; margin-bottom: 5px; color: #fff; }}
            .link-desc {{ font-size: 0.85rem; color: var(--text-sub); }}
            
            .knowledge-paper {{ background-color: #1a1a1a; border: 1px solid var(--border); border-radius: 8px; padding: 40px; min-height: 400px; color: #ddd; }}
            .knowledge-paper h3 {{ border-bottom: 2px solid var(--accent); padding-bottom: 10px; margin-top: 0; }}
            
            details {{ margin-top: 15px; border-top: 1px solid #333; padding-top: 10px; }}
            summary {{ cursor: pointer; color: #777; font-size: 0.85rem; }}
            .abstract-text {{ margin-top: 10px; font-size: 0.9rem; color: #bbb; text-align: justify; line-height: 1.6; word-break: keep-all; }}
            
            .arxiv-ack {{ text-align: center; margin-top: 40px; margin-bottom: 20px; color: #444; font-size: 0.8rem; font-family: monospace; padding-top: 20px; border-top: 1px solid #222; }}
            
            footer {{ margin-top: auto; padding-top: 20px; }}

            @media (max-width: 480px) {{
                body {{ padding: 15px; }}
                .card-grid {{ grid-template-columns: 1fr; }} 
                .tabs {{ justify-content: flex-start; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>AI 정보</h1>
                <div class="timestamp">Updated: {now_kst}</div>
            </header>

            <nav class="tabs">
                <button class="tab-btn" onclick="openTab('Knowledge')">지식</button>
                <button class="tab-btn active" onclick="openTab('News')">뉴스</button>
                <button class="tab-btn" onclick="openTab('Papers')">논문</button>
                <button class="tab-btn" onclick="openTab('Labs')">연구소</button>
                <button class="tab-btn" onclick="openTab('Conferences')">학회</button>
                <button class="tab-btn" onclick="openTab('Others')">기타</button>
            </nav>

            <main>
                {content_html}
            </main>

            <footer></footer>
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
                    var txt = btns[i].innerText;
                    
                    if (
                        (tabName === 'News' && txt.includes('뉴스')) ||
                        (tabName === 'Papers' && txt.includes('논문')) ||
                        (tabName === 'Labs' && txt.includes('연구소')) ||
                        (tabName === 'Conferences' && txt.includes('학회')) ||
                        (tabName === 'Knowledge' && txt.includes('지식')) ||
                        (tabName === 'Others' && txt.includes('기타'))
                    ) {{
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
        if 'Labs' in rss_data:
            process_translation(rss_data['Labs'], ['title'])
            
        if paper_data:
            print(" - 논문(Papers) 초록 번역 중...")
            process_translation(paper_data, ['title', 'abstract'])

    html_out = create_html(rss_data, paper_data, CONFERENCE_LINKS, OTHER_LINKS, KNOWLEDGE_CONTENT)
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)
        
    print("index.html Generated Successfully.")
