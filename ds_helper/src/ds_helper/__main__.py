"""
Main entry point for the DS Helper package.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ds_helper.scraper.cli import main as scraper_main, parse_args as scraper_parse_args, run_with_args as scraper_run
from ds_helper.processor.cli import main as processor_main, parse_args as processor_parse_args, run_with_args as processor_run
from ds_helper.query.cli import main as query_main, parse_args as query_parse_args, run_with_args as query_run
from ds_helper.web.cli import main as web_main, parse_args as web_parse_args, run_with_args as web_run
from ds_helper.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_RAW_DIR = os.path.join(DEFAULT_DATA_DIR, "raw")
DEFAULT_PROCESSED_DIR = os.path.join(DEFAULT_DATA_DIR, "processed")
DEFAULT_VECTORDB_PATH = os.path.join(os.getcwd(), "ds_helper_vectordb")
DEFAULT_OPENBIS_URL = "https://openbis.readthedocs.io/en/20.10.0-11/"
DEFAULT_WIKIJS_URL = "https://datastore.bam.de/en/home"  # This should be configured by user
DEFAULT_MAX_PAGES = None  # No limit - scrape all pages


def run_full_pipeline():
    """Run the full pipeline: scrape both sources, process, ingest to vector DB."""
    logger.info("Starting the full DS Helper pipeline...")

    # Create directories if they don't exist
    os.makedirs(DEFAULT_RAW_DIR, exist_ok=True)
    os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

    # Save original argv
    original_argv = sys.argv.copy()

    try:
        # Step 1: Scrape ReadTheDocs (openBIS)
        logger.info(f"Scraping openBIS documentation from {DEFAULT_OPENBIS_URL}...")
        rtd_output_dir = os.path.join(DEFAULT_RAW_DIR, "openbis")
        os.makedirs(rtd_output_dir, exist_ok=True)
        
        scraper_args_list = [
            "readthedocs",
            "--url", DEFAULT_OPENBIS_URL,
            "--output", rtd_output_dir,
            "--verbose"
        ]
        if DEFAULT_MAX_PAGES is not None:
            scraper_args_list += ["--max-pages", str(DEFAULT_MAX_PAGES)]

        scraper_args = scraper_parse_args(scraper_args_list)
        
        scraper_result = scraper_run(scraper_args)
        if scraper_result != 0:
            logger.error("ReadTheDocs scraping failed.")
            return scraper_result

        # Step 2: Scrape Wiki.js (if URL is provided)
        if DEFAULT_WIKIJS_URL == "https://datastore.bam.de/en/home":
            logger.info(f"Scraping Wiki.js from {DEFAULT_WIKIJS_URL}...")
            wiki_output_dir = os.path.join(DEFAULT_RAW_DIR, "wikijs")
            os.makedirs(wiki_output_dir, exist_ok=True)
            
            scraper_args_list = [
                "readthedocs",
                "--url", DEFAULT_OPENBIS_URL,
                "--output", rtd_output_dir,
                "--verbose"
            ]
            if DEFAULT_MAX_PAGES is not None:
                scraper_args_list += ["--max-pages", str(DEFAULT_MAX_PAGES)]

            scraper_args = scraper_parse_args(scraper_args_list)
            
            scraper_result = scraper_run(scraper_args)
            if scraper_result != 0:
                logger.error("Wiki.js scraping failed.")
                return scraper_result
        else:
            logger.warning("Wiki.js URL not correct. Skipping Wiki.js scraping.")
            logger.warning("Please set the Wiki.js URL in the configuration or use individual commands.")

        # Step 3: Process all scraped content
        logger.info(f"Processing scraped content from {DEFAULT_RAW_DIR}...")
        processor_args = processor_parse_args([
            "--input", DEFAULT_RAW_DIR,
            "--output", DEFAULT_PROCESSED_DIR,
            "--min-chunk-size", "50",  # Reduce minimum chunk size
            "--max-chunk-size", "800"  # Reduce maximum chunk size
        ])

        processor_result = processor_run(processor_args)
        if processor_result != 0:
            logger.error("Processing failed.")
            return processor_result

        # Step 4: Ingest into vector database
        logger.info("Ingesting processed content into vector database...")
        chunks_file = os.path.join(DEFAULT_PROCESSED_DIR, "chunks.json")
        
        if os.path.exists(chunks_file):
            # Import and run the ingestion script
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
            from ingest_to_vectordb import main as ingest_main
            
            # Temporarily modify sys.argv for the ingestion script
            sys.argv = [
                "ingest_to_vectordb",
                "--chunks-file", chunks_file,
                "--db-path", DEFAULT_VECTORDB_PATH,
                "--reset"  # Reset the database for fresh ingestion
            ]
            
            ingest_result = ingest_main()
            if ingest_result != 0:
                logger.error("Vector database ingestion failed.")
                return ingest_result
        else:
            logger.error(f"Chunks file not found: {chunks_file}")
            return 1

        # Step 5: Start the query interface
        logger.info("Starting DS Helper query interface...")
        query_args = query_parse_args([
            "--db-path", DEFAULT_VECTORDB_PATH
        ])

        return query_run(query_args)

    finally:
        # Restore original argv
        sys.argv = original_argv


def check_vectordb_exists():
    """Check if vector database exists."""
    db_path = Path(DEFAULT_VECTORDB_PATH)
    return db_path.exists() and any(db_path.iterdir())


def auto_mode():
    """Automatically determine what to do based on data availability."""
    setup_logging(logging.INFO)

    if check_vectordb_exists():
        logger.info("Found existing vector database. Checking if it has data...")

        # Check if the database actually has data
        try:
            from ds_helper.utils.vector_db import DSHelperVectorDB
            vector_db = DSHelperVectorDB(db_path=DEFAULT_VECTORDB_PATH)
            stats = vector_db.get_collection_stats()
            vector_db.close()

            if stats.get('total_chunks', 0) > 0:
                logger.info(f"Database has {stats['total_chunks']} chunks. Starting DS Helper...")
                # Save original argv
                original_argv = sys.argv.copy()

                # Temporarily modify sys.argv for the query module
                sys.argv = ["ds_helper"]

                # Import and use the parse_args function from query.cli
                query_args = query_parse_args([
                    "--db-path", DEFAULT_VECTORDB_PATH
                ])

                # Restore original argv
                sys.argv = original_argv

                # Call query_run with the parsed args
                return query_run(query_args)
            else:
                logger.info("Database exists but is empty. Running full pipeline...")
                return run_full_pipeline()

        except Exception as e:
            logger.warning(f"Error checking database: {e}. Running full pipeline...")
            return run_full_pipeline()
    else:
        logger.info("No vector database found. Running full pipeline...")
        return run_full_pipeline()


def main():
    """Main entry point for the package."""
    # Check if no arguments were provided
    if len(sys.argv) == 1:
        return auto_mode()

    parser = argparse.ArgumentParser(
        description="DS Helper - A RAG-based chatbot for openBIS and Data Store documentation.",
        prog="ds_helper"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Scrape content from documentation sites")
    scraper_subparsers = scraper_parser.add_subparsers(dest="scraper_type", help="Type of scraper")
    
    # ReadTheDocs scraper
    rtd_parser = scraper_subparsers.add_parser("readthedocs", help="Scrape ReadTheDocs site")
    rtd_parser.add_argument("--url", required=True, help="The base URL of the ReadtheDocs site")
    rtd_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    rtd_parser.add_argument("--version", help="The specific version to scrape (e.g., 'en/latest')")
    rtd_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    rtd_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    rtd_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Wiki.js scraper
    wiki_parser = scraper_subparsers.add_parser("wikijs", help="Scrape Wiki.js site")
    wiki_parser.add_argument("--url", required=True, help="The base URL of the Wiki.js site")
    wiki_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    wiki_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    wiki_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    wiki_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Processor command
    processor_parser = subparsers.add_parser("process", help="Process content for RAG")
    processor_parser.add_argument("--input", required=True, help="The directory containing the scraped content")
    processor_parser.add_argument("--output", required=True, help="The directory to save the processed content to")
    processor_parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    processor_parser.add_argument("--min-chunk-size", type=int, default=100, help="The minimum size of a chunk in characters")
    processor_parser.add_argument("--max-chunk-size", type=int, default=1000, help="The maximum size of a chunk in characters")
    processor_parser.add_argument("--chunk-overlap", type=int, default=50, help="The overlap between chunks in characters")
    processor_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query using RAG with vector database")
    query_parser.add_argument("--db-path", default=DEFAULT_VECTORDB_PATH, help="Path to the ChromaDB database directory")
    query_parser.add_argument("--collection-name", default="ds_helper_docs", help="Name of the ChromaDB collection")
    query_parser.add_argument("--model", default="gpt-oss:20b", help="The Ollama model to use for chat")
    query_parser.add_argument("--top-k", type=int, default=5, help="The number of chunks to retrieve")
    query_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    query_parser.add_argument("--stats", action="store_true", help="Show database statistics and exit")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest processed chunks into vector database")
    ingest_parser.add_argument("--chunks-file", required=True, help="Path to the JSON file containing processed chunks")
    ingest_parser.add_argument("--db-path", default=DEFAULT_VECTORDB_PATH, help="Path to the ChromaDB database directory")
    ingest_parser.add_argument("--collection-name", default="ds_helper_docs", help="Name of the ChromaDB collection")
    ingest_parser.add_argument("--reset", action="store_true", help="Reset the collection before ingesting")
    ingest_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Web command
    web_parser = subparsers.add_parser("web", help="Run the web interface")
    web_parser.add_argument("--db-path", default=DEFAULT_VECTORDB_PATH, help="Path to the ChromaDB database directory")
    web_parser.add_argument("--collection-name", default="ds_helper_docs", help="Name of the ChromaDB collection")
    web_parser.add_argument("--model", default="gpt-oss:20b", help="The Ollama model to use for chat")
    web_parser.add_argument("--host", default="127.0.0.1", help="The host to run the web interface on")
    web_parser.add_argument("--port", type=int, default=5000, help="The port to run the web interface on")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    web_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Auto command (hidden, for internal use)
    subparsers.add_parser("auto", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "scrape":
        return scraper_main()
    elif args.command == "process":
        return processor_main()
    elif args.command == "query":
        return query_main()
    elif args.command == "web":
        return web_main()
    elif args.command == "ingest":
        # Import and run the ingestion script
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
            from ingest_to_vectordb import main as ingest_main
            return ingest_main()
        except ImportError as e:
            logger.error(f"Could not import ingestion script: {e}")
            return 1
    elif args.command == "auto":
        return auto_mode()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
