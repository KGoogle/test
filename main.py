import datetime
import os
import time
import re
import pytz
import json
from time import mktime
import google.generativeai as genai
import feedparser
import requests
import arxiv
from email.utils import parsedate_to_datetime
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = 'gemini-2.5-flash-lite' 

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

RSS_SOURCES = {
    "Labs": {
        "Google(DeepMind)": "https://blog.google/technology/research/rss/",
        "Microsoft(MSR)": "https://www.microsoft.com/en-us/research/feed/",
        "OpenAI": "https://openai.com/news/rss.xml",
        "NVIDIA": "https://blogs.nvidia.com/blog/category/generative-ai/feed/",
        "UC Berkeley(BAIR)": "https://bair.berkeley.edu/blog/feed.xml",
        "CMU(SCS)": "https://www.cs.cmu.edu/news/feed",
        "MIT(CSAIL)": "https://news.mit.edu/rss/topic/artificial-intelligence2"
    },
    "News": {
        "AI 타임스": "https://cdn.aitimes.com/rss/gn_rss_allArticle.xml",
        "인공지능신문": "https://www.aitimes.kr/rss/S1N2.xml",
        "AI Matters": "https://aimatters.co.kr/category/news-report/ai-report/feed/",
        "GeekNews(뉴스)": "https://feeds.feedburner.com/geeknews-feed"
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
    {"name": "Meta(MSL)", "url": "https://ai.meta.com/research/", "desc": "RSS 제공 안함"},
    {"name": "Anthropic", "url": "https://www.anthropic.com/research", "desc": "RSS 제공 안함"},
    {"name": "xAI", "url": "https://x.ai/news", "desc": "RSS 제공 안함"},
    {"name": "Stanford(SAIL)", "url": "https://ai.stanford.edu/blog/", "desc": "RSS 제공 안함"},
    {"name": "MIT(CSAIL)", "url": "https://www.csail.mit.edu/research/?category=Groups", "desc": "RSS 제공 안함"}
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
"""
def load_and_generate_knowledge():
    db_path = 'knowledge_db.json'
    
    if not os.path.exists(db_path):
        print(f"Warning: {db_path} not found.")
        return KNOWLEDGE_CONTENT + "<h3>Knowledge DB not found.</h3>"

    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading knowledge DB: {e}")
        return KNOWLEDGE_CONTENT + f"<h3>Error loading Knowledge DB: {e}</h3>"

    html_parts = [KNOWLEDGE_CONTENT, '<h3>AI History Timeline</h3>', '<div class="timeline-container">']
    
    for item in data:
        summary = item.get('summary', 'No Title')
        events = item.get('events', [])
        
        events_html = '<ul class="event-list">'
        for event in events:
            title = event.get('title', '')
            desc = event.get('desc', '')
            events_html += f'<li><strong>{title}</strong><br>{desc}</li>'
        events_html += '</ul>'

        item_html = f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <details>
                <summary>{summary}</summary>
                <div class="timeline-content">
                    {events_html}
                </div>
            </details>
        </div>
        """
        html_parts.append(item_html)
    
    html_parts.append('</div>')
    return "".join(html_parts)

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

def translate_batch_with_gemini(texts, field_type='text'):
    if not GOOGLE_API_KEY or not texts:
        return texts

    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.1
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            MODEL_NAME, 
            generation_config=generation_config,
            safety_settings=safety_settings 
        )
        
        separator = " ||| "
        formatted_text = separator.join(texts)

        if field_type == 'title':
            prompt = f"""
            Role: Professional AI Researcher and Translator.
            Task: Translate the following list of AI research paper titles or blog headlines into Korean.
            
            Rules:
            1. Output ONLY the translated titles.
            2. Use the separator "{separator}" between titles exactly as in the input.
            3. Do NOT add any list numbers (e.g., 1., 2.) at the beginning of lines.
            4. Keep specific model names (e.g., Gemini, GPT-4, Llama-3) and technical acronyms (LLM, RAG) in English.
            5. Keep the translation concise and professional.
            
            Input Text:
            {formatted_text}
            """
        else:
            prompt = f"""
            Role: Professional AI Researcher.
            Task: Translate the following list of academic abstracts into natural, professional Korean.
            
            Rules:
            1. Output ONLY the translated text blocks.
            2. Use the separator "{separator}" between abstracts exactly as in the input.
            3. Maintain technical accuracy. 
            4. Keep standard technical terms in English if the Korean translation is awkward.
            
            Input Text:
            {formatted_text}
            """
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        translated_list = [t.strip() for t in result_text.split('|||')]

        if len(translated_list) != len(texts):
            print(f"Warning: Batch size mismatch. Sent {len(texts)}, got {len(translated_list)}. Using original text.")
            return texts
            
        return translated_list
    
    except Exception as e:
        print(f"Translation Error: {e}")
        return texts

def process_translation(data_list, fields):
    if not GOOGLE_API_KEY:
        return

    batch_size = 5
    print(f"Translating {len(data_list)} items in batches of {batch_size}...")

    for field in fields:
        field_type = 'title' if 'title' in field.lower() else 'text'

        for i in range(0, len(data_list), batch_size):
            batch_items = data_list[i : i + batch_size]
            
            texts_to_translate = [item.get(field, "") for item in batch_items]
            
            valid_items_with_index = [(idx, t) for idx, t in enumerate(texts_to_translate) if t]
            
            if valid_items_with_index:
                valid_indices = [v[0] for v in valid_items_with_index]
                valid_texts = [v[1] for v in valid_items_with_index]

                translated_texts = translate_batch_with_gemini(valid_texts, field_type)
                
                for k, original_idx in enumerate(valid_indices):
                    if k < len(translated_texts):
                        batch_items[original_idx][field] = translated_texts[k]
            
            print(f" - {field}: Batch {i//batch_size + 1} done.")
            time.sleep(5)

def create_html(rss_data, paper_data, conf_links, other_links, knowledge_content):
    now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    # [삭제 예정_START]
    DEV_Note_WIDGET = """
    <style>
        .dev-btn {
            position: absolute; top: 15px; right: 20px;
            background: rgba(255, 255, 255, 0.1); border: 1px solid #444; color: #888;
            padding: 5px 10px; border-radius: 5px; cursor: pointer; font-size: 12px;
            transition: all 0.2s; z-index: 9999; text-decoration: none;
        }
        .dev-btn:hover { background: #333; color: #fff; border-color: #666; }
        
        .dev-modal {
            display: none; position: fixed; z-index: 10000; left: 0; top: 0;
            width: 100%; height: 100%; overflow: auto;
            background-color: rgba(0,0,0,0.8);
            justify-content: center; align-items: center;
        }
        
        .dev-modal-content {
            background-color: #1a1a1a; margin: auto; padding: 30px;
            border: 1px solid #444; border-radius: 12px;
            width: 80%; max-width: 500px;
            color: #e0e0e0; box-shadow: 0 5px 20px rgba(0,0,0,0.5);
            position: relative; animation: fadeIn 0.2s;
        }
        
        .dev-close {
            position: absolute; top: 10px; right: 20px; color: #aaa;
            font-size: 28px; font-weight: bold; cursor: pointer;
        }
        .dev-close:hover { color: #fff; }
        
        .dev-list { list-style-type: none; padding: 0; margin-top: 15px; text-align: left; }
        .dev-list li { 
            padding: 8px 0; border-bottom: 1px solid #333; font-size: 0.95rem; color: #ccc; 
            display: flex; align-items: start;
        }
        .dev-list li::before { content: "▫️"; margin-right: 10px; }
        .dev-list li strong { color: #fff; margin-right: 5px; }
        .dev-list li:last-child { border-bottom: none; }
        
        @media (max-width: 480px) {
            .dev-btn { top: 10px; right: 10px; padding: 4px 8px; font-size: 11px; }
        }
    </style>

    <button class="dev-btn" onclick="document.getElementById('devModal').style.display='flex'">📝 Dev Note</button>

    <div id="devModal" class="dev-modal" onclick="if(event.target == this) this.style.display='none'">
        <div class="dev-modal-content">
            <span class="dev-close" onclick="document.getElementById('devModal').style.display='none'">&times;</span>
            <h2 style="margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 10px; font-size: 1.5rem;">To-Do</h2>
            
            <ul class="dev-list">
                <li><strong>[지식]</strong> 정보 수집 후 정리</li>
                <li><strong>[뉴스]</strong> 해외 뉴스 추가 고려</li>
                <li><strong>[논문]</strong> 논문 수집 범위 수정 필요</li>
                <li><strong>[연구소]</strong> 연구소 추가 및 정보 변경</li>
                <li><strong>[학회]</strong> 보류</li>
                <li><strong>[기타]</strong> 보류</li>
            </ul>
        </div>
    </div>
    """
    # [삭제 예정_END]
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
                    <summary>Abstract</summary>
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
        <!-- [삭제 예정_START] -->
        {DEV_Note_WIDGET}
        <!-- [삭제 예정_END] -->
        
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
    knowledge_content = load_and_generate_knowledge()

    if GOOGLE_API_KEY:
        if 'Labs' in rss_data:
            process_translation(rss_data['Labs'], ['title'])
            
        if paper_data:
            print(" - 논문(Papers) 초록 번역 중...")
            process_translation(paper_data, ['title', 'abstract'])

    html_out = create_html(rss_data, paper_data, CONFERENCE_LINKS, OTHER_LINKS, knowledge_content)
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)
        
    print("index.html Generated Successfully.")
