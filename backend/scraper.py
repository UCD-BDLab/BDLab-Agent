import requests
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time

class JsonScrape:
    """ json scrapper"""
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "JsonScrape/1.0"})

    def fetch_json(self, url):
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

class HtmlScrape:
    """single domain crawler for html pages"""
    def __init__(self, start_url, max_pages=30):
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.max_pages = max_pages

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HtmlScrape/1.0"})

        self.seen = set()
        self.json_scraper = JsonScrape(self.session)

    def fetch(self, url):
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    def get_links(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all('a', href=True):
            full = urljoin(base_url, a['href'])
            parsed = urlparse(full)

            if parsed.scheme.startswith('http') and parsed.netloc == self.domain:
                links.append(full)
                
        return links

    def crawl(self):
        queue = [self.start_url]
        output = {}

        while queue and len(self.seen) < self.max_pages:
            url = queue.pop(0)
            if url in self.seen:
                continue

            print("Scraping:", url)
            html = self.fetch(url)
            if not html:
                continue

            self.seen.add(url)
            soup = BeautifulSoup(html, "html.parser")

            
            data = {"url": url, "title": None, "text": None, "images": [], "links": [] }

            if soup.title and soup.title.string:
                data["title"] = soup.title.string.strip()

            data["text"] = soup.get_text(separator=" ", strip=True)
            for img in soup.find_all('img', src=True):
                src = img['src']
                full_src = urljoin(url, src)
                data["images"].append(full_src)

            if url.lower().endswith(".html"):
                json_url = url[:-5] + ".json"
                jd = self.json_scraper.fetch_json(json_url)

                if jd is not None:
                    data["json_equivalent"] = {json_url: jd}

            new_links = self.get_links(html, url)
            for link in new_links:
                data["links"].append(link)

                if link not in self.seen and link not in queue:
                    queue.append(link)

            output[url] = data
            time.sleep(0.3)

        return output

if __name__ == "__main__":
    
    start = "https://cse.ucdenver.edu/~bdlab/"
    html_scraper = HtmlScrape(start, max_pages=20)
    result = html_scraper.crawl()
    
    with open("scrapped.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
