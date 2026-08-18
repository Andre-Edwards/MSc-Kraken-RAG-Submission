from __future__ import annotations

import argparse
import json

from app.core.config import settings
from app.core.database import bump_corpus_version, get_connection, get_corpus_index_versions
from app.services.web_scraper import crawl_web_pages, expand_sitemap_urls, fetch_robots_sitemaps


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl a bounded set of public web pages into the local corpus.")
    parser.add_argument("--seed", action="append", dest="seeds", default=[], help="Seed URL. Repeat for multiple URLs.")
    parser.add_argument("--root-url", default="https://www.kraken.com", help="Root URL used when reading /robots.txt.")
    parser.add_argument(
        "--from-robots-sitemaps",
        action="store_true",
        help="Read Sitemap entries from the root URL's robots.txt and crawl URLs declared there.",
    )
    parser.add_argument(
        "--sitemap",
        action="append",
        dest="sitemaps",
        default=[],
        help="Sitemap URL to expand into crawl seeds. Repeat for multiple sitemaps.",
    )
    parser.add_argument(
        "--sitemap-url-limit",
        type=int,
        default=5000,
        help="Maximum number of page URLs to collect from sitemaps before crawling.",
    )
    parser.add_argument(
        "--allowed-domain",
        action="append",
        dest="allowed_domains",
        default=[],
        help="Allowed domain. Defaults to configured Kraken domains.",
    )
    parser.add_argument("--max-pages", type=int, default=settings.web_crawl_default_max_pages)
    parser.add_argument("--max-depth", type=int, default=settings.web_crawl_default_max_depth)
    parser.add_argument("--delay-seconds", type=float, default=settings.web_crawl_delay_seconds)
    parser.add_argument("--timeout-seconds", type=float, default=settings.web_crawl_timeout_seconds)
    parser.add_argument("--user-agent", default=settings.web_crawl_user_agent)
    parser.add_argument(
        "--include-path",
        action="append",
        dest="include_paths",
        default=[],
        help="Regex that URL paths must match. Repeat for multiple allowed path patterns.",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        dest="exclude_paths",
        default=[],
        help="Regex that URL paths must not match. Repeat for multiple blocked path patterns.",
    )
    args = parser.parse_args()

    sitemap_urls = list(args.sitemaps)
    sitemap_result: dict | None = None
    if args.from_robots_sitemaps:
        sitemap_urls.extend(
            fetch_robots_sitemaps(
                root_url=args.root_url,
                user_agent=args.user_agent,
                timeout_seconds=args.timeout_seconds,
            )
        )

    seeds = list(args.seeds)
    if sitemap_urls:
        sitemap_result = expand_sitemap_urls(
            sitemap_urls=_dedupe_urls(sitemap_urls),
            allowed_domains=args.allowed_domains or None,
            max_urls=args.sitemap_url_limit,
            include_path_patterns=args.include_paths or None,
            exclude_path_patterns=args.exclude_paths or None,
            delay_seconds=args.delay_seconds,
            user_agent=args.user_agent,
            timeout_seconds=args.timeout_seconds,
        )
        seeds.extend(sitemap_result["urls"])

    seeds = _dedupe_urls(seeds)
    if not seeds:
        raise SystemExit("No crawl seeds provided. Use --seed, --sitemap, or --from-robots-sitemaps.")

    result = crawl_web_pages(
        seed_urls=seeds,
        allowed_domains=args.allowed_domains or None,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        user_agent=args.user_agent,
        timeout_seconds=args.timeout_seconds,
        include_path_patterns=args.include_paths or None,
        exclude_path_patterns=args.exclude_paths or None,
    )
    if sitemap_result:
        result["sitemap_discovery"] = {
            "root_url": args.root_url,
            "sitemaps": _dedupe_urls(sitemap_urls),
            "sitemaps_visited": sitemap_result["sitemaps_visited"],
            "page_urls_discovered": len(sitemap_result["urls"]),
            "skipped": sitemap_result["skipped"],
        }
    if result.get("pages_saved", 0) > 0:
        result["corpus_version"] = bump_corpus_version()
    else:
        with get_connection() as conn:
            result["corpus_version"] = get_corpus_index_versions(conn)["corpus_version"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
