#!/usr/bin/env python3
"""CLI interface for Sougou dictionary spider."""

if __name__ == "__main__":
    import argparse
    import logging
    import os
    from pathlib import Path

    from sougou_dict_spider import SougouDictSpider

    parser = argparse.ArgumentParser(
        description="A Sougou dictionary spider.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        help="The directory to save Sougou dictionaries.\n"
        f"Default: {SougouDictSpider.DEFAULT_SAVE_PATH}",
        metavar="DIR",
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        help="List of category indices to be downloaded.\n"
        "Categories are not separated to their subcategories.\n"
        "Special category 0 is for dictionaries "
        "that do not belong to any categories.\n"
        "Download all categories (including 0) by default.",
        metavar="CATEGORY",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        help="List of dictionary indices to exclude downloading.\n"
        "Default (nonexistent dictionaries): "
        f"{', '.join(sorted(SougouDictSpider.DEFAULT_EXCLUDE_LIST))}",
        metavar="DICTIONARY",
    )
    parser.add_argument(
        "--concurrent-downloads",
        "-j",
        type=int,
        help="Set the number of parallel downloads.\n"
        "Default: min(32, (os.process_cpu_count() or 1) + 4) "
        f"({min(32, (os.process_cpu_count() or 1) + 4)})",
        metavar="N",
    )
    parser.add_argument(
        "--max-retries",
        "-m",
        type=int,
        help="Set the maximum number of retries.\n"
        f"Default: {SougouDictSpider.DEFAULT_MAX_RETRIES}",
        metavar="N",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        help=f"Set timeout in seconds.\nDefault: {SougouDictSpider.DEFAULT_TIMEOUT}",
        metavar="SEC",
    )
    parser.add_argument(
        "--force",
        "-f",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-download of existing files.\nDefault: False",
    )
    parser.add_argument(
        "--debug",
        "-D",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output debug info.\nDefault: False",
    )
    parser.add_argument(
        "--error",
        "-E",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output only error info.\nIgnored if --debug is specified."
        "\nDefault: False",
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s:%(message)s",
        level=logging.DEBUG
        if args.debug
        else logging.ERROR
        if args.error
        else logging.INFO,
    )
    with SougouDictSpider(
        args.categories,
        args.directory,
        args.exclude,
        args.concurrent_downloads,
        args.max_retries,
        args.timeout,
        force=args.force,
    ) as dict_spider:
        dict_spider.download_dicts()
