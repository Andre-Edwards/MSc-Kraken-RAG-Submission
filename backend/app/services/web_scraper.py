from __future__ import annotations

import json
import re
import ssl
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from app.core.config import settings
from app.models import PageRecord
from app.services.text_cleaning import basic_clean, clean_pages


WEB_PAGES_FILE = "web_pages.jsonl"

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

SSL_CONTEXT = ssl.create_default_context()
SITEMAP_LIMIT = 5000


@dataclass
class WebPageRecord:
    url: str
    final_url: str
    title: str
    text: str
    status_code: int
    content_type: str
    fetched_at: str


class ReadableHtmlParser(HTMLParser):
    SKIP_TAGS = {
        "button",
        "canvas",
        "footer",
        "form",
        "header",
        "nav",
        "noscript",
        "script",
        "select",
        "style",
        "svg",
    }
    BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_dict = {key.lower(): value for key, value in attrs if value is not None}
        if tag in self.SKIP_TAGS:
            self._skip_stack.append(tag)
            return
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = attr_dict.get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        self.text_parts.append(text)
        self.text_parts.append(" ")

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        text = "".join(self.text_parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return basic_clean(text)


def _canonical_url(url: str) -> str:
    cleaned, _fragment = urldefrag(url.strip())
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        return ""
    path = parsed.path or "/"
    return parsed._replace(path=path, query=parsed.query, fragment="").geturl().rstrip("/")


def _domain_allowed(url: str, allowed_domains: set[str]) -> bool:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def _looks_like_html_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    blocked_exts = (
        ".avi",
        ".css",
        ".csv",
        ".doc",
        ".docx",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".json",
        ".mp3",
        ".mp4",
        ".pdf",
        ".png",
        ".svg",
        ".webp",
        ".xls",
        ".xlsx",
        ".zip",
    )
    return not path.endswith(blocked_exts)


def _matches_path_filters(
    url: str,
    include_path_patterns: list[str] | None = None,
    exclude_path_patterns: list[str] | None = None,
) -> bool:
    path = urlparse(url).path or "/"
    if include_path_patterns and not any(re.search(pattern, path, re.I) for pattern in include_path_patterns):
        return False
    if exclude_path_patterns and any(re.search(pattern, path, re.I) for pattern in exclude_path_patterns):
        return False
    return True


def _sleep_if_needed(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


class RobotsCache:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self._cache: dict[str, robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._cache.get(origin)
        if parser is None:
            parser = robotparser.RobotFileParser()
            parser.set_url(urljoin(origin, "/robots.txt"))
            try:
                request = Request(parser.url, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=settings.web_crawl_timeout_seconds, context=SSL_CONTEXT) as response:
                    content = response.read().decode("utf-8", errors="replace").splitlines()
                parser.parse(content)
            except Exception:
                return False
            self._cache[origin] = parser
        return parser.can_fetch(self.user_agent, url)


def _fetch_html(url: str, user_agent: str, timeout: float) -> tuple[str, str, int, str]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html"})
    with urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        status = getattr(response, "status", 200)
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()
        raw = response.read()
    if "html" not in content_type.lower():
        raise ValueError(f"Skipping non-HTML response: {content_type}")
    encoding_match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    encoding = encoding_match.group(1) if encoding_match else "utf-8"
    html = raw.decode(encoding, errors="replace")
    return html, final_url, status, content_type


def _fetch_text(url: str, user_agent: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def fetch_robots_sitemaps(
    root_url: str,
    user_agent: str | None = None,
    timeout_seconds: float | None = None,
) -> list[str]:
    user_agent = user_agent or settings.web_crawl_user_agent
    timeout_seconds = settings.web_crawl_timeout_seconds if timeout_seconds is None else timeout_seconds
    parsed = urlparse(root_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid root URL: {root_url}")
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    text = _fetch_text(robots_url, user_agent, timeout_seconds)
    sitemaps: list[str] = []
    for line in text.splitlines():
        if line.lower().startswith("sitemap:"):
            value = line.split(":", 1)[1].strip()
            if value:
                sitemaps.append(value)
    return sitemaps


def _xml_urls(xml_text: str) -> tuple[list[str], list[str]]:
    root = ElementTree.fromstring(xml_text.encode("utf-8"))
    locs = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.lower().endswith("loc") and (element.text or "").strip()
    ]
    sitemap_urls = [url for url in locs if urlparse(url).path.lower().endswith(".xml")]
    page_urls = [url for url in locs if not urlparse(url).path.lower().endswith(".xml")]
    return sitemap_urls, page_urls


def expand_sitemap_urls(
    sitemap_urls: list[str],
    allowed_domains: list[str] | None = None,
    max_urls: int = SITEMAP_LIMIT,
    include_path_patterns: list[str] | None = None,
    exclude_path_patterns: list[str] | None = None,
    delay_seconds: float | None = None,
    user_agent: str | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    user_agent = user_agent or settings.web_crawl_user_agent
    timeout_seconds = settings.web_crawl_timeout_seconds if timeout_seconds is None else timeout_seconds
    delay_seconds = settings.web_crawl_delay_seconds if delay_seconds is None else delay_seconds
    allowed = {
        domain.strip().lower()
        for domain in (allowed_domains or settings.get_default_allowed_web_domains())
        if domain.strip()
    }

    queue = list(dict.fromkeys(sitemap_urls))
    visited_sitemaps: set[str] = set()
    page_urls: list[str] = []
    skipped: list[dict[str, str]] = []

    while queue and len(page_urls) < max_urls:
        sitemap_url = queue.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)
        if not _domain_allowed(sitemap_url, allowed):
            skipped.append({"url": sitemap_url, "reason": "sitemap_outside_allowed_domains"})
            continue
        try:
            xml_text = _fetch_text(sitemap_url, user_agent, timeout_seconds)
            child_sitemaps, child_pages = _xml_urls(xml_text)
        except (HTTPError, URLError, TimeoutError, ElementTree.ParseError, ValueError) as exc:
            skipped.append({"url": sitemap_url, "reason": str(exc)[:200]})
            _sleep_if_needed(delay_seconds)
            continue
        _sleep_if_needed(delay_seconds)
        for child_sitemap in child_sitemaps:
            if child_sitemap not in visited_sitemaps:
                queue.append(child_sitemap)
        for page_url in child_pages:
            canonical = _canonical_url(page_url)
            if (
                canonical
                and _domain_allowed(canonical, allowed)
                and _looks_like_html_url(canonical)
                and _matches_path_filters(canonical, include_path_patterns, exclude_path_patterns)
                and canonical not in page_urls
            ):
                page_urls.append(canonical)
                if len(page_urls) >= max_urls:
                    break

    return {
        "urls": page_urls[:max_urls],
        "sitemaps_visited": len(visited_sitemaps),
        "skipped": skipped[:50],
    }


def _record_to_page(record: WebPageRecord, page_num: int = 1) -> PageRecord:
    title = record.title or urlparse(record.final_url).path.strip("/") or record.final_url
    parsed = urlparse(record.final_url)
    url_key = (parsed.netloc + parsed.path).strip("/") or record.final_url
    doc_id = f"Web - {title[:90]} - {url_key[:90]}"
    file_name = f"Web - {title[:140]}"
    return PageRecord(
        doc_id=doc_id,
        file_name=file_name,
        source_path=record.final_url,
        page_num=page_num,
        text=record.text,
        clean_text=basic_clean(record.text),
        likely_scanned=False,
    )


def save_web_records(records: Iterable[WebPageRecord], output_dir: Path | None = None, merge: bool = True) -> Path:
    target_dir = output_dir or settings.resolve_web_corpus_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / WEB_PAGES_FILE
    all_records: dict[str, WebPageRecord] = {}
    if merge and path.exists():
        for existing in load_web_records(target_dir):
            all_records[existing.final_url] = existing
    for record in records:
        all_records[record.final_url] = record
    with path.open("w", encoding="utf-8") as f:
        for record in sorted(all_records.values(), key=lambda item: item.final_url):
            f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
    return path


def delete_web_record(url: str, web_dir: Path | None = None) -> WebPageRecord | None:
    target = url.strip().rstrip("/")
    records = load_web_records(web_dir)
    kept: list[WebPageRecord] = []
    deleted: WebPageRecord | None = None
    for record in records:
        record_urls = {record.url.strip().rstrip("/"), record.final_url.strip().rstrip("/")}
        if target in record_urls:
            deleted = record
            continue
        kept.append(record)
    if deleted is not None:
        save_web_records(kept, output_dir=web_dir, merge=False)
    return deleted


def load_web_records(web_dir: Path | None = None) -> list[WebPageRecord]:
    target_dir = web_dir or settings.resolve_web_corpus_dir()
    path = target_dir / WEB_PAGES_FILE
    if not path.exists():
        return []
    records: list[WebPageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        lines = list(f)
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(WebPageRecord(**data))
    return records


def load_web_corpus(web_dir: Path | None = None) -> list[PageRecord]:
    records = load_web_records(web_dir)
    pages = [_record_to_page(record) for record in records if record.text.strip()]
    return clean_pages(pages)


def _duplicate_normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _word_shingles(text: str, size: int = 8) -> set[tuple[str, ...]]:
    tokens = _duplicate_normalize(text).split()
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)}


def _jaccard_similarity(first: set[tuple[str, ...]], second: set[tuple[str, ...]]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def detect_duplicate_web_records(
    web_dir: Path | None = None,
    similarity_threshold: float = 0.86,
) -> dict:
    records = load_web_records(web_dir)
    exact_groups: dict[str, list[WebPageRecord]] = {}
    normalized_by_url: dict[str, str] = {}
    for record in records:
        normalized = _duplicate_normalize(record.text)
        if not normalized:
            continue
        normalized_by_url[record.final_url] = normalized
        exact_groups.setdefault(normalized, []).append(record)

    exact_duplicate_groups = [
        {
            "type": "exact",
            "similarity": 1.0,
            "pages": [
                {
                    "title": item.title,
                    "url": item.final_url,
                    "bytes": len(item.text.encode("utf-8")),
                }
                for item in group
            ],
        }
        for group in exact_groups.values()
        if len(group) > 1
    ]

    shingled = [(record, _word_shingles(record.text)) for record in records if record.text.strip()]
    near_pairs = []
    for left_index, (left_record, left_shingles) in enumerate(shingled):
        for right_record, right_shingles in shingled[left_index + 1 :]:
            if normalized_by_url.get(left_record.final_url) == normalized_by_url.get(right_record.final_url):
                continue
            similarity = _jaccard_similarity(left_shingles, right_shingles)
            if similarity >= similarity_threshold:
                near_pairs.append(
                    {
                        "type": "near",
                        "similarity": round(similarity, 3),
                        "pages": [
                            {
                                "title": left_record.title,
                                "url": left_record.final_url,
                                "bytes": len(left_record.text.encode("utf-8")),
                            },
                            {
                                "title": right_record.title,
                                "url": right_record.final_url,
                                "bytes": len(right_record.text.encode("utf-8")),
                            },
                        ],
                    }
                )

    duplicate_groups = exact_duplicate_groups + sorted(
        near_pairs,
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return {
        "total_pages": len(records),
        "similarity_threshold": similarity_threshold,
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
    }


def crawl_web_pages(
    seed_urls: list[str],
    allowed_domains: list[str] | None = None,
    max_pages: int | None = None,
    max_depth: int | None = None,
    delay_seconds: float | None = None,
    user_agent: str | None = None,
    timeout_seconds: float | None = None,
    include_path_patterns: list[str] | None = None,
    exclude_path_patterns: list[str] | None = None,
) -> dict:
    allowed = {
        domain.strip().lower()
        for domain in (allowed_domains or settings.get_default_allowed_web_domains())
        if domain.strip()
    }
    user_agent = user_agent or settings.web_crawl_user_agent
    max_pages = max_pages or settings.web_crawl_default_max_pages
    max_depth = settings.web_crawl_default_max_depth if max_depth is None else max_depth
    delay_seconds = settings.web_crawl_delay_seconds if delay_seconds is None else delay_seconds
    timeout_seconds = settings.web_crawl_timeout_seconds if timeout_seconds is None else timeout_seconds

    if not seed_urls:
        raise ValueError("At least one seed URL is required.")
    if not allowed:
        raise ValueError("At least one allowed domain is required.")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")
    if max_depth < 0:
        raise ValueError("max_depth must be 0 or greater.")

    queue: list[tuple[str, int]] = []
    for seed in seed_urls:
        canonical = _canonical_url(seed)
        if canonical and _domain_allowed(canonical, allowed) and _matches_path_filters(
            canonical,
            include_path_patterns,
            exclude_path_patterns,
        ):
            queue.append((canonical, 0))

    robots = RobotsCache(user_agent)
    visited: set[str] = set()
    records: list[WebPageRecord] = []
    skipped: list[dict[str, str]] = []

    while queue and len(records) < max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if not _domain_allowed(url, allowed):
            skipped.append({"url": url, "reason": "outside_allowed_domains"})
            continue
        if not _looks_like_html_url(url):
            skipped.append({"url": url, "reason": "non_html_extension"})
            continue
        if not _matches_path_filters(url, include_path_patterns, exclude_path_patterns):
            skipped.append({"url": url, "reason": "path_filter_excluded"})
            continue
        if not robots.can_fetch(url):
            skipped.append({"url": url, "reason": "blocked_by_robots_txt"})
            _sleep_if_needed(delay_seconds)
            continue

        try:
            html, final_url, status, content_type = _fetch_html(url, user_agent, timeout_seconds)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            skipped.append({"url": url, "reason": str(exc)[:200]})
            _sleep_if_needed(delay_seconds)
            continue

        parser = ReadableHtmlParser(final_url)
        parser.feed(html)
        text = parser.text
        if len(text.split()) < 80:
            skipped.append({"url": url, "reason": "too_little_text"})
        else:
            records.append(
                WebPageRecord(
                    url=url,
                    final_url=_canonical_url(final_url) or final_url,
                    title=parser.title or final_url,
                    text=text,
                    status_code=int(status),
                    content_type=content_type,
                    fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
            )

        if depth < max_depth:
            for link in parser.links:
                canonical_link = _canonical_url(link)
                if (
                    canonical_link
                    and canonical_link not in visited
                    and _domain_allowed(canonical_link, allowed)
                    and _looks_like_html_url(canonical_link)
                    and _matches_path_filters(canonical_link, include_path_patterns, exclude_path_patterns)
                ):
                    queue.append((canonical_link, depth + 1))

        _sleep_if_needed(delay_seconds)

    output_path = save_web_records(records)
    return {
        "ok": True,
        "pages_saved": len(records),
        "visited": len(visited),
        "skipped": skipped[:50],
        "output_path": str(output_path),
        "allowed_domains": sorted(allowed),
        "seed_count": len(seed_urls),
        "seed_urls": seed_urls[:100],
        "seed_urls_truncated": len(seed_urls) > 100,
        "max_pages": max_pages,
        "max_depth": max_depth,
    }
