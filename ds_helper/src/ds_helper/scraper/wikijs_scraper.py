#!/usr/bin/env python3
"""
Wiki.js Scraper

A module for scraping content from Wiki.js documentation sites.
This module extracts all textual content from a Wiki.js site and saves it
to text files for use in downstream RAG (Retrieval Augmented Generation) pipelines.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)


class WikiJSParser:
    """Parser for Wiki.js HTML content."""

    def __init__(self):
        """Initialize the parser."""
        self.content_selectors = [
            "div.v-content__wrap",  # Wiki.js main content wrapper
            "div.page-content",  # Alternative content div
            "main.v-content",  # Main content element
            "article",  # HTML5 article element
            "div.content",  # Generic content div
            "div.wiki-content",  # Wiki-specific content div
        ]
        
        self.ignore_selectors = [
            "nav",  # Navigation
            "footer",  # Footer
            "div.v-navigation-drawer",  # Side navigation
            "div.v-app-bar",  # Top app bar
            "div.v-toolbar",  # Toolbar
            "div.breadcrumbs",  # Breadcrumbs
            "div.page-actions",  # Page action buttons
            "div.page-meta",  # Page metadata
            "div.toc",  # Table of contents
            "div.v-card__actions",  # Card actions
            "div.edit-fab",  # Edit floating action button
        ]

    def extract_content(self, html_content: str, url: str) -> Dict[str, str]:
        """
        Extract the main content from a Wiki.js HTML page.
        
        Args:
            html_content: The HTML content of the page
            url: The URL of the page
            
        Returns:
            A dictionary containing the title and content of the page
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Extract the title - Wiki.js often uses different title structures
        title = ""
        
        # Try different title extraction methods
        if soup.title:
            title = soup.title.string.strip()
            # Clean up common Wiki.js title patterns
            title = re.sub(r'\s*\|\s*.*$', '', title)  # Remove site name after |
            title = re.sub(r'\s*-\s*.*$', '', title)   # Remove site name after -
        
        # Try to find page title in content
        if not title or title == "":
            title_selectors = [
                "h1.page-title",
                "h1.v-card__title", 
                "h1",
                ".page-title",
                ".v-card__title"
            ]
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text().strip()
                    break
        
        # Find the main content
        content_element = None
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            logger.warning(f"Could not find main content in {url}")
            return {"title": title, "content": "", "url": url}
        
        # Remove elements to ignore
        for selector in self.ignore_selectors:
            for element in content_element.select(selector):
                element.decompose()
        
        # Extract text content
        content = self._extract_text_with_structure(content_element)
        
        return {"title": title, "content": content, "url": url}
    
    def _extract_text_with_structure(self, element) -> str:
        """
        Extract text from an element while preserving some structure.
        
        Args:
            element: The BeautifulSoup element to extract text from
            
        Returns:
            The extracted text with some structure preserved
        """
        if element.name in ["pre", "code"]:
            # For code blocks, preserve formatting
            return f"\n```\n{element.get_text()}\n```\n"
        
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # For headings, add appropriate markdown
            level = int(element.name[1])
            return f"\n{'#' * level} {element.get_text().strip()}\n"
        
        if element.name == "p":
            # For paragraphs, ensure they're separated by newlines
            return f"\n{element.get_text().strip()}\n"
        
        if element.name == "li":
            # For list items, add a bullet point
            return f"- {element.get_text().strip()}\n"
        
        if element.name == "table":
            # For tables, we'll just extract the text for now
            return f"\n{element.get_text().strip()}\n"
        
        # Recursively process child elements
        if hasattr(element, "children"):
            result = ""
            for child in element.children:
                if hasattr(child, "name"):
                    result += self._extract_text_with_structure(child)
                elif child.string and child.string.strip():
                    result += child.string
            return result
        
        # If it's just a string, return it
        return element.string if element.string else ""


class WikiJSScraper:
    """Scraper for Wiki.js documentation sites."""

    def __init__(
        self, 
        base_url: str, 
        output_dir: str, 
        delay: float = 0.5,
        max_pages: Optional[int] = None
    ):
        """
        Initialize the scraper.
        
        Args:
            base_url: The base URL of the Wiki.js site
            output_dir: The directory to save the scraped content to
            delay: The delay between requests in seconds
            max_pages: The maximum number of pages to scrape (None for unlimited)
        """
        self.base_url = self._sanitize_url(base_url)
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.max_pages = max_pages
        
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: List[str] = [self.base_url]
        self.parser = WikiJSParser()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse the base URL to get the domain
        parsed_url = urlparse(self.base_url)
        self.domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    def _sanitize_url(self, url: str) -> str:
        """
        Sanitize the URL to ensure it's properly formatted.
        
        Args:
            url: The URL to sanitize
            
        Returns:
            The sanitized URL
        """
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        # Remove trailing slash
        url = url.rstrip("/")
        
        return url
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid for scraping.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is valid, False otherwise
        """
        # Skip if already visited
        if url in self.visited_urls:
            return False
        
        # Skip if not from the same domain
        if not url.startswith(self.domain):
            return False
        
        # Skip common non-content URLs
        skip_patterns = [
            r'/api/',
            r'/login',
            r'/register',
            r'/admin',
            r'/edit',
            r'/history',
            r'/source',
            r'/raw',
            r'\.pdf$',
            r'\.zip$',
            r'\.tar\.gz$',
            r'\.jpg$',
            r'\.png$',
            r'\.gif$',
            r'\.css$',
            r'\.js$',
            r'#',  # Skip anchor links
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def _save_content(self, content: Dict[str, str]) -> None:
        """
        Save the content to a file.
        
        Args:
            content: The content dictionary containing title, content, and URL
        """
        # Create a filename from the URL
        url_path = urlparse(content["url"]).path
        filename = url_path.strip("/").replace("/", "_")
        
        # If the filename is empty, use the domain
        if not filename:
            filename = "index"
        
        # Remove any invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Add the .txt extension
        filename = f"{filename}.txt"
        
        # Create the full path
        filepath = self.output_dir / filename
        
        # Write the content to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {content['title']}\n")
            f.write(f"URL: {content['url']}\n")
            f.write(f"Source: datastore\n")  # Add source metadata
            f.write(f"---\n\n")
            f.write(content["content"])
        
        logger.info(f"Saved content to {filepath}")
    
    def scrape(self) -> None:
        """
        Scrape the Wiki.js site.
        """
        logger.info(f"Starting to scrape {self.base_url}")
        
        pages_scraped = 0
        
        while self.urls_to_visit and (self.max_pages is None or pages_scraped < self.max_pages):
            # Get the next URL to visit
            url = self.urls_to_visit.pop(0)
            
            # Skip if we've already visited this URL
            if url in self.visited_urls:
                continue
            
            logger.info(f"Scraping {url}")
            
            try:
                # Make the request
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Mark the URL as visited
                self.visited_urls.add(url)
                
                # Extract the content
                content = self.parser.extract_content(response.text, url)
                
                # Save the content
                self._save_content(content)
                
                # Find links to other pages
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    
                    # Skip empty links
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute URLs
                    if not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)
                    
                    # Check if the URL is valid
                    if self._is_valid_url(href):
                        self.urls_to_visit.append(href)
                
                pages_scraped += 1
                
                # Add delay between requests
                if self.delay > 0:
                    time.sleep(self.delay)
                
            except requests.RequestException as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error scraping {url}: {e}")
                continue
        
        logger.info(f"Finished scraping. Scraped {pages_scraped} pages.")
