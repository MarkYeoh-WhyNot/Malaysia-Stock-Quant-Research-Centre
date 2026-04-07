"""
I3investor Scraper — scrapes klse.i3investor.com for research articles,
news headlines, dividend announcements, article content, and forum posts.
"""
import logging
import re
import time
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://klse.i3investor.com/",
}

_BASE = "https://klse.i3investor.com"

# Known KLCI bursa codes → ticker mapping for mention detection
_KLCI_CODES = {
    "1155", "1295", "1023", "5347", "5183", "5225", "8869", "6947",
    "6012", "1066", "1961", "5285", "5819", "3182", "4715", "4863",
    "4707", "4065", "6033", "3816", "5168", "2445", "7277", "1015",
    "4197", "1082", "4677", "5398", "5296",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _safe_date(raw: str) -> str:
    """Normalise various date strings to YYYY-MM-DD. Returns raw on failure."""
    raw = raw.strip()
    for fmt in ("%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%b %d, %Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw


def _extract_ticker_mentions(text: str) -> list:
    """Find Bursa stock codes mentioned in a block of text."""
    found = []
    for code in _KLCI_CODES:
        if re.search(r"\b" + code + r"\b", text):
            found.append(f"{code}.KL")
    return found


def _strip_html(html: str) -> str:
    """Remove HTML tags, scripts, styles and collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


class I3investorScraper:
    """
    Scraper for klse.i3investor.com.

    All methods return plain Python dicts / lists — no external dependencies
    beyond requests and BeautifulSoup4 (both already installed).
    """

    def __init__(self):
        self._session = _make_session()

    # ------------------------------------------------------------------
    # Internal fetch helper
    # ------------------------------------------------------------------

    def _get(self, url: str, timeout: int = 20) -> Optional[BeautifulSoup]:
        try:
            resp = self._session.get(url, timeout=timeout)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"i3investor fetch failed ({url}): {e}")
            return None

    # ------------------------------------------------------------------
    # 1. Research articles
    # ------------------------------------------------------------------

    def get_research_articles(self, max_articles: int = 20) -> list:
        """
        Scrape research/analyst articles from i3investor.

        Returns list of dicts:
          title, author, brokerage, url, date, tickers (list)
        """
        url = f"{_BASE}/web/headline/blog?type=research"
        soup = self._get(url)
        if not soup:
            return []

        articles = []
        # i3investor renders article summaries in list items or divs with class patterns
        # Try multiple selectors in order of specificity
        items = (
            soup.select("div.news-item")
            or soup.select("div.article-item")
            or soup.select("li.list-group-item")
            or soup.select("div.row div.col-md-12")
        )

        if not items:
            # Fallback: grab all <a> tags with /web/blog/ in href
            items = []
            for a in soup.find_all("a", href=re.compile(r"/web/blog/")):
                parent = a.find_parent(["div", "li", "tr"])
                if parent and parent not in items:
                    items.append(parent)

        for item in items[:max_articles]:
            try:
                # Title + URL
                link = item.find("a", href=re.compile(r"/web/blog/|/web/headline/"))
                if not link:
                    continue
                title = link.get_text(strip=True)
                href  = link.get("href", "")
                full_url = href if href.startswith("http") else f"{_BASE}{href}"

                # Date
                date_tag = item.find(class_=re.compile(r"date|time", re.I)) or item.find("small")
                date_str = date_tag.get_text(strip=True) if date_tag else ""
                date_str = _safe_date(date_str)

                # Author / brokerage — look for spans or small tags after the title
                author = ""
                brokerage = ""
                for span in item.find_all(["span", "small", "div"]):
                    t = span.get_text(strip=True)
                    brokerages = ["RHB", "Kenanga", "Maybank", "CIMB", "PublicBank",
                                  "AmInvest", "PhillipCapital", "Hong Leong", "UOB",
                                  "Affin", "Alliance", "BIMB"]
                    for b in brokerages:
                        if b.lower() in t.lower():
                            brokerage = b
                    if not author and t and len(t) < 60 and t != title:
                        author = t

                # Ticker mentions in title + author text
                combined = f"{title} {author}"
                tickers = _extract_ticker_mentions(combined)

                articles.append({
                    "title":     title,
                    "author":    author,
                    "brokerage": brokerage,
                    "url":       full_url,
                    "date":      date_str,
                    "tickers":   tickers,
                })
            except Exception as e:
                logger.debug(f"Article parse error: {e}")

        logger.info(f"i3investor: scraped {len(articles)} research articles")
        return articles

    # ------------------------------------------------------------------
    # 2. News headlines
    # ------------------------------------------------------------------

    def get_news_headlines(self, max_items: int = 30) -> list:
        """
        Scrape latest news headlines from i3investor.

        Returns list of dicts:
          headline, date, url, tickers (list)
        """
        url = f"{_BASE}/web/headline/news"
        soup = self._get(url)
        if not soup:
            return []

        headlines = []
        # News items are typically in a table or list
        rows = (
            soup.select("table.table tr")
            or soup.select("div.news-item")
            or soup.select("li.list-group-item")
        )

        if not rows:
            rows = []
            for a in soup.find_all("a", href=re.compile(r"/web/news/|/web/headline/")):
                parent = a.find_parent(["tr", "div", "li"])
                if parent and parent not in rows:
                    rows.append(parent)

        for row in rows[:max_items]:
            try:
                link = row.find("a")
                if not link:
                    continue
                headline = link.get_text(strip=True)
                if not headline or len(headline) < 5:
                    continue
                href = link.get("href", "")
                full_url = href if href.startswith("http") else f"{_BASE}{href}"

                date_tag = row.find(class_=re.compile(r"date|time", re.I)) or row.find("small")
                date_str = date_tag.get_text(strip=True) if date_tag else ""
                date_str = _safe_date(date_str)

                tickers = _extract_ticker_mentions(headline)

                headlines.append({
                    "headline": headline,
                    "date":     date_str,
                    "url":      full_url,
                    "tickers":  tickers,
                })
            except Exception as e:
                logger.debug(f"Headline parse error: {e}")

        logger.info(f"i3investor: scraped {len(headlines)} news headlines")
        return headlines

    # ------------------------------------------------------------------
    # 3. Dividend announcements
    # ------------------------------------------------------------------

    def get_dividend_announcements(self, max_items: int = 20) -> list:
        """
        Scrape dividend announcements by filtering news headlines for
        'Dividend' or 'Ex Date' keywords.

        Returns list of dicts:
          company, headline, dividend_amount, ex_date, url, date
        """
        headlines = self.get_news_headlines(max_items=100)

        dividends = []
        for h in headlines:
            text = h["headline"].lower()
            if "dividend" not in text and "ex date" not in text and "ex-date" not in text:
                continue

            # Try to extract dividend amount (e.g. "5.0 sen", "RM 0.05")
            amount = ""
            amt_match = re.search(
                r"(\d+\.?\d*)\s*(sen|cent|rm|%)", h["headline"], re.IGNORECASE
            )
            if amt_match:
                amount = f"{amt_match.group(1)} {amt_match.group(2)}"

            # Try to extract ex-date
            ex_date = ""
            date_match = re.search(
                r"ex[- ]date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w{3}\s+\d{4})",
                h["headline"], re.IGNORECASE
            )
            if date_match:
                ex_date = _safe_date(date_match.group(1))

            # Company name: typically at start of headline before dash or colon
            company = ""
            parts = re.split(r"[-–:]", h["headline"], maxsplit=1)
            if parts:
                company = parts[0].strip()

            dividends.append({
                "company":         company or h["headline"][:40],
                "headline":        h["headline"],
                "dividend_amount": amount,
                "ex_date":         ex_date or h["date"],
                "url":             h["url"],
                "date":            h["date"],
                "tickers":         h["tickers"],
            })

            if len(dividends) >= max_items:
                break

        logger.info(f"i3investor: found {len(dividends)} dividend announcements")
        return dividends

    # ------------------------------------------------------------------
    # 4. Article full content
    # ------------------------------------------------------------------

    def get_article_content(self, url: str) -> str:
        """
        Fetch and return the clean plain-text content of a single article URL.
        Strips navigation, ads, sidebars, and HTML tags.
        """
        soup = self._get(url)
        if not soup:
            return ""

        # Remove boilerplate sections
        for tag in soup.find_all(["nav", "header", "footer", "aside", "script",
                                   "style", "form", "iframe"]):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile(
            r"nav|menu|sidebar|ad|banner|comment|share|social|footer|header|related",
            re.IGNORECASE
        )):
            tag.decompose()

        # Try to find the main article body
        article_body = (
            soup.find("div", class_=re.compile(r"article-content|post-content|entry-content|blog-content", re.I))
            or soup.find("div", id=re.compile(r"content|article|post", re.I))
            or soup.find("article")
            or soup.find("main")
        )

        if article_body:
            text = article_body.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        clean = "\n".join(lines)
        logger.info(f"i3investor: fetched article content ({len(clean)} chars) from {url}")
        return clean

    # ------------------------------------------------------------------
    # 5. Forum posts for a stock
    # ------------------------------------------------------------------

    def get_forum_posts(self, stock_code: str, max_posts: int = 10) -> list:
        """
        Scrape forum posts for a specific Bursa stock code.

        stock_code: Bursa 4-digit code (e.g. "1155") or Yahoo ticker ("1155.KL")
        Returns list of dicts: content, date, author, url
        """
        # Normalise to bare code
        code = stock_code.replace(".KL", "").strip()
        url = f"{_BASE}/web/forum/forum-thread/{code}"
        soup = self._get(url)
        if not soup:
            return []

        time.sleep(1)

        posts = []
        # Forum posts are usually in .panel, .post, or table rows
        post_items = (
            soup.select("div.panel-body")
            or soup.select("div.post-content")
            or soup.select("div.comment-body")
            or soup.select("td.message")
        )

        if not post_items:
            post_items = soup.find_all("div", class_=re.compile(r"post|comment|reply|message", re.I))

        for item in post_items[:max_posts]:
            try:
                content = item.get_text(separator=" ", strip=True)
                if not content or len(content) < 10:
                    continue

                # Try to find date in parent or sibling
                parent = item.find_parent(["div", "tr"])
                date_tag = None
                if parent:
                    date_tag = parent.find(class_=re.compile(r"date|time", re.I)) or parent.find("small")
                date_str = date_tag.get_text(strip=True) if date_tag else ""
                date_str = _safe_date(date_str)

                # Author
                author_tag = None
                if parent:
                    author_tag = parent.find(class_=re.compile(r"author|user|name", re.I))
                author = author_tag.get_text(strip=True) if author_tag else ""

                posts.append({
                    "content": content[:500],  # Cap post length
                    "date":    date_str,
                    "author":  author,
                    "url":     url,
                })
            except Exception as e:
                logger.debug(f"Forum post parse error: {e}")

        logger.info(f"i3investor: scraped {len(posts)} forum posts for {code}")
        return posts
