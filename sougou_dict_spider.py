"""A spider for downloading Sougou dictionaries."""

import itertools
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast
from urllib.parse import urljoin

import requests
import tenacity
from bs4 import BeautifulSoup

from queue_thread_pool_executor import QueueThreadPoolExecutor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from concurrent.futures import Future
    from types import TracebackType

    from bs4.element import Tag

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class SougouDictSpider:
    """A spider for downloading Sougou dictionaries."""

    DEFAULT_SAVE_PATH = Path("sougou_dict")
    DEFAULT_EXCLUDE_LIST = frozenset(("2775", "15946", "176476"))
    DEFAULT_MAX_RETRIES = 20
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_FORCE = True

    def __init__(
        self,
        categories: Iterable[str] | None = None,
        save_path: Path | None = None,
        exclude_list: Iterable[str] | None = None,
        concurrent_downloads: int | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        *,
        force: bool = DEFAULT_FORCE,
    ) -> None:
        """
        Initialize a new SougouDictSpider instance.

        Args:
            categories: Iterable of category indices to be downloaded.
            save_path: Directory to save dictionaries.
            exclude_list: Iterable of dictionary indices to exclude.
            concurrent_downloads: Number of parallel downloads.
            max_retries: Maximum number of retries for HTTP requests.
            timeout: Timeout for HTTP requests in seconds.
            force: If True, force re-download of existing files.
        """
        self._categories = frozenset(categories) if categories is not None else None
        self._save_path = save_path or SougouDictSpider.DEFAULT_SAVE_PATH
        self._exclude_list = (
            frozenset(exclude_list)
            if exclude_list is not None
            else SougouDictSpider.DEFAULT_EXCLUDE_LIST
        )
        self._max_retries = (
            SougouDictSpider.DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        )
        self._timeout = SougouDictSpider.DEFAULT_TIMEOUT if timeout is None else timeout
        self._force = force
        self._stats = {"downloaded": 0, "skipped": 0, "failed": 0}
        self._thread_local = threading.local()
        self._sessions: list[requests.Session] = []
        self._executor = QueueThreadPoolExecutor(concurrent_downloads)
        self._lock = threading.Lock()
        self._futures: list[Future[object]] = []
        self._shutdown = False

    def _submit[**P, T](
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        with self._lock:
            self._futures.append(self._executor.submit(fn, *args, **kwargs))

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            with self._lock:
                self._sessions.append(session)
            self._thread_local.session = session
        return session

    def _download_html(self, url: str) -> requests.Response:

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self._max_retries + 1),
            wait=tenacity.wait_exponential(multiplier=1, max=60),
            retry=tenacity.retry_if_exception_type(requests.RequestException),
            before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
        )
        def _attempt() -> requests.Response:
            response = self._get_session().get(url, timeout=self._timeout)
            response.raise_for_status()
            if not response.content:
                msg = f"Empty content for {url}"
                raise requests.RequestException(msg)
            return response

        try:
            return _attempt()
        except tenacity.RetryError:
            log.exception("Failed to download %s", url)
            raise

    def _download(self, name: str, url: str, category_path: Path) -> None:
        file_path = category_path / name
        if not self._force and file_path.is_file():
            log.warning("%s already exists, skipping...", file_path)
            with self._lock:
                self._stats["skipped"] += 1
            return
        try:
            file_path.write_bytes(self._download_html(url).content)
        except Exception as e:
            with self._lock:
                self._stats["failed"] += 1
            if isinstance(e, OSError):
                log.exception("Failed to save to %s", file_path)
            raise
        else:
            with self._lock:
                self._stats["downloaded"] += 1
            log.info("%s downloaded successfully.", name)

    @staticmethod
    def _sanitize(raw: str) -> str:
        return raw.translate(
            str.maketrans({
                "<": "-",
                ">": "-",
                ":": "-",
                '"': "-",
                "/": "-",
                "\\": "-",
                "|": "-",
                "?": "-",
                "*": "-",
            })
        )

    def _download_page(self, page_url: str, category_path: Path) -> None:
        for dict_td in BeautifulSoup(
            self._download_html(page_url).text, "html.parser"
        ).select("div.dict_detail_block"):
            dict_td_title = cast("Tag", dict_td.select_one("div.detail_title a"))
            dict_td_id = cast("str", dict_td_title.get("href")).rpartition("/")[-1]
            if dict_td_id not in self._exclude_list:
                self._submit(
                    self._download,
                    SougouDictSpider._sanitize(cast("str", dict_td_title.string))
                    + "_"
                    + dict_td_id
                    + ".scel",
                    urljoin(
                        page_url,
                        cast(
                            "str",
                            cast("Tag", dict_td.select_one("div.dict_dl_btn a")).get(
                                "href"
                            ),
                        ),
                    ),
                    category_path,
                )

    def _download_category(
        self, category: str, *, sub_of_category_167: bool = False
    ) -> None:
        if category == "0":
            self._download_category_0()
            return
        if category == "167":
            self._download_category_167()
            return
        category_url = f"https://pinyin.sogou.com/dict/cate/index/{category}"
        soup = BeautifulSoup(self._download_html(category_url).text, "html.parser")
        if not sub_of_category_167:
            category_path = self._save_path / (
                cast("Tag", soup.select_one("title"))
                .get_text(strip=True)
                .partition("_")[0]
                + f"_{category}"
            )
            category_path.mkdir(exist_ok=True)
        else:
            category_path = self._save_path / "城市信息大全_167"
        pages = soup.select("div#dict_page_list a")
        try:
            page_n = int(pages[-2].get_text(strip=True)) + 1
        except IndexError, TypeError, ValueError:
            page_n = 2
        for page in range(1, page_n):
            self._submit(
                self._download_page, f"{category_url}/default/{page}", category_path
            )

    def _download_category_167(self) -> None:
        soup = BeautifulSoup(
            self._download_html("https://pinyin.sogou.com/dict/cate/index/180").text,
            "html.parser",
        )
        (self._save_path / "城市信息大全_167").mkdir(exist_ok=True)
        for category_td in soup.select("div.citylistcate"):
            self._submit(
                self._download_category,
                cast(
                    "str", cast("Tag", category_td.select_one("a")).get("href")
                ).rpartition("/")[-1],
                sub_of_category_167=True,
            )

    def _download_category_0(self) -> None:
        index_url = "https://pinyin.sogou.com/dict/detail/index/4"
        soup = BeautifulSoup(self._download_html(index_url).text, "html.parser")
        category_path = self._save_path / "未分类_0"
        category_path.mkdir(exist_ok=True)
        self._submit(
            self._download,
            "网络流行新词【官方推荐】_4.scel",
            "https://pinyin.sogou.com/d/dict/download_cell.php?id=4&name=网络流行新词【官方推荐】",
            category_path,
        )
        for dict_td in soup.select("div.rcmd_dict"):
            dict_td_title = cast("Tag", dict_td.select_one("div.rcmd_dict_title a"))
            self._submit(
                self._download,
                SougouDictSpider._sanitize(cast("str", dict_td_title.string))
                + "_"
                + cast("str", dict_td_title.get("href")).rpartition("/")[-1]
                + ".scel",
                urljoin(
                    index_url,
                    cast(
                        "str",
                        cast("Tag", dict_td.select_one("div.rcmd_dict_dl_btn a")).get(
                            "href"
                        ),
                    ),
                ),
                category_path,
            )

    def _download_category_list(self) -> frozenset[str]:
        if self._categories is not None:
            return self._categories
        return frozenset(
            itertools.chain(
                "0",
                (
                    cast("str", cast("Tag", category.select_one("a")).get("href"))
                    .partition("?")[0]
                    .rpartition("/")[-1]
                    for category in BeautifulSoup(
                        self._download_html("https://pinyin.sogou.com/dict/").text,
                        "html.parser",
                    ).select("div.dict_category_list_title")
                ),
            )
        )

    def download_dicts(self) -> None:
        """
        Download dictionaries.

        Raises:
            RuntimeError: If the spider has been shut down.
        """
        if self._shutdown:
            msg = "Cannot download dictionaries after shutdown."
            raise RuntimeError(msg)
        self._save_path.mkdir(parents=True, exist_ok=True)
        for category in self._download_category_list():
            self._submit(self._download_category, category)

    def _report_stats(self) -> int:
        log.info("")
        log.info("---- Dictionary Download Summary ----")
        log.info("downloaded=%d", self._stats["downloaded"])
        log.info("skipped=%d", self._stats["skipped"])
        log.info("failed=%d", self._stats["failed"])
        log.info("")
        log.info("---- Detailed Exception Summary ----")
        failures = 0
        for future in self._futures:
            exception = future.exception()
            if exception is not None:
                failures += 1
                log.error("[%d] %s", failures, exception)
        if not failures:
            log.info("No exceptions occurred during the download process.")
        return failures

    def shutdown(self) -> None:
        """
        Shutdown the spider.

        Raises:
            RuntimeError: If there were failures during the download process.
        """
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=True)
        for s in self._sessions:
            s.close()
        failures = self._report_stats()
        if failures:
            msg = f"Downloading finished with {failures} failures."
            raise RuntimeError(msg)

    def __enter__(self) -> Self:
        """
        Enter the runtime context related to this object.

        Returns:
            Self: The SougouDictSpider instance itself.
        """
        return self

    def __exit__(
        self,
        typ: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """
        Exit the runtime context and clean up resources.

        Args:
            typ: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.

        Returns:
            False to indicate that exceptions should not be suppressed.
        """
        self.shutdown()
        return False
