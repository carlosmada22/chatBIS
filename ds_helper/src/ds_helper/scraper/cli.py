#!/usr/bin/env python3
"""
Command-line interface for DS Helper scrapers.
"""

import argparse
import logging
import sys

from .readthedocs_scraper import ReadTheDocsScraper
from .wikijs_scraper import WikiJSScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape content from documentation sites.")
    
    subparsers = parser.add_subparsers(dest="scraper_type", help="Type of scraper to use")
    
    # ReadTheDocs scraper
    rtd_parser = subparsers.add_parser("readthedocs", help="Scrape ReadTheDocs site")
    rtd_parser.add_argument("--url", required=True, help="The base URL of the ReadtheDocs site")
    rtd_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    rtd_parser.add_argument("--version", help="The specific version to scrape (e.g., 'en/latest')")
    rtd_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    rtd_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    rtd_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Wiki.js scraper
    wiki_parser = subparsers.add_parser("wikijs", help="Scrape Wiki.js site")
    wiki_parser.add_argument("--url", required=True, help="The base URL of the Wiki.js site")
    wiki_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    wiki_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    wiki_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    wiki_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the scraper with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        if args.scraper_type == "readthedocs":
            # Create and run the ReadTheDocs scraper
            scraper = ReadTheDocsScraper(
                base_url=args.url,
                output_dir=args.output,
                target_version=args.version,
                delay=args.delay,
                max_pages=args.max_pages
            )
        elif args.scraper_type == "wikijs":
            # Create and run the Wiki.js scraper
            scraper = WikiJSScraper(
                base_url=args.url,
                output_dir=args.output,
                delay=args.delay,
                max_pages=args.max_pages
            )
        else:
            logger.error("Invalid scraper type. Use 'readthedocs' or 'wikijs'.")
            return 1

        scraper.scrape()
        return 0
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    if not args.scraper_type:
        print("Please specify a scraper type: 'readthedocs' or 'wikijs'")
        return 1
    
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
