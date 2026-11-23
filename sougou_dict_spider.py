"""A spider for downloading Sougou dictionaries."""

import itertools
import logging
import os
import threading
from concurrent.futures import Executor
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Self
from urllib.parse import urljoin
lazy from collections.abc import Callable, Iterable
lazy from concurrent.futures import Future
lazy from types import TracebackType

import requests
import tenacity
from bs4 import BeautifulSoup, Tag
from typeguard import check_type

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class DownloadError(RuntimeError):
    """Exception raised if any exceptions occurred during the download process."""


class SougouDictSpider:
    """A spider for downloading Sougou dictionaries."""

    DEFAULT_SAVE_PATH = Path("sougou_dict")
    DEFAULT_EXCLUDE_LIST = frozenset(("2775", "15946", "176476"))
    DEFAULT_MAX_RETRIES = 20
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_CONCURRENT_DOWNLOADS = (os.process_cpu_count() or 1) + 4
    DEFAULT_FORCE = False

    class _ThreadPoolExecutor(Executor):
        def __init__(self, max_workers: int | None = None) -> None:
            if max_workers is None:
                max_workers = SougouDictSpider.DEFAULT_CONCURRENT_DOWNLOADS
            if max_workers <= 0:
                msg = "max_workers must be greater than 0"
                raise ValueError(msg)
            self._max_workers = max_workers
            self._task_queue: Queue[
                tuple[
                    Callable[..., object],
                    tuple[object, ...],
                    dict[str, object],
                    Future[Any],
                ]
                | None
            ] = Queue()
            for _ in range(self._max_workers):
                Thread(target=self._executor, daemon=True).start()

        def shutdown(
            self,
            wait: bool = True,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
            *,
            cancel_futures: bool = False,
        ) -> None:
            if cancel_futures:
                raise NotImplementedError
            if wait:
                self._task_queue.join()
            for _ in range(self._max_workers):
                self._task_queue.put(None)
                if wait:
                    self._task_queue.join()

        def submit[**P, T](
            self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
        ) -> Future[T]:
            future: Future[T] = Future()
            self._task_queue.put((fn, args, kwargs, future))
            return future

        def _executor(self) -> None:
            while True:
                task = self._task_queue.get()
                if task is None:
                    self._task_queue.task_done()
                    return
                fn, args, kwargs, future = task
                try:
                    if not future.set_running_or_notify_cancel():
                        continue
                    result = fn(*args, **kwargs)
                except BaseException as exc:  # ruff: ignore[blind-except]
                    future.set_exception(exc)
                else:
                    future.set_result(result)
                finally:
                    self._task_queue.task_done()

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
        self._count = 0
        self._thread_local = threading.local()
        self._sessions: list[requests.Session] = []
        self._executor = SougouDictSpider._ThreadPoolExecutor(concurrent_downloads)
        self._lock = threading.Lock()
        self._exceptions: list[BaseException] = []
        self._shutdown = False

    def _record_exception[T](self, future: Future[T]) -> None:
        exception = future.exception()
        if exception is not None:
            with self._lock:
                self._exceptions.append(exception)

    def _submit[**P, T](
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._executor.submit(fn, *args, **kwargs).add_done_callback(
            self._record_exception
        )

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
            with self._lock:
                self._count += 1
            log.warning("%s already exists, skipping...", file_path)
            return
        try:
            file_path.write_bytes(self._download_html(url).content)
        except Exception as e:
            if isinstance(e, OSError):
                log.exception("Failed to save to %s", file_path)
            raise
        else:
            with self._lock:
                self._count += 1
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
            dict_td_title = check_type(dict_td.select_one("div.detail_title a"), Tag)
            dict_td_id = check_type(dict_td_title.get("href"), str).rpartition("/")[-1]
            if dict_td_id not in self._exclude_list:
                self._submit(
                    self._download,
                    SougouDictSpider._sanitize(check_type(dict_td_title.string, str))
                    + "_"
                    + dict_td_id
                    + ".scel",
                    urljoin(
                        page_url,
                        check_type(
                            check_type(
                                dict_td.select_one("div.dict_dl_btn a"), Tag
                            ).get("href"),
                            str,
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
                check_type(soup.select_one("title"), Tag)
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
        except IndexError, ValueError:
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
                check_type(
                    check_type(category_td.select_one("a"), Tag).get("href"), str
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
            dict_td_title = check_type(dict_td.select_one("div.rcmd_dict_title a"), Tag)
            self._submit(
                self._download,
                SougouDictSpider._sanitize(check_type(dict_td_title.string, str))
                + "_"
                + check_type(dict_td_title.get("href"), str).rpartition("/")[-1]
                + ".scel",
                urljoin(
                    index_url,
                    check_type(
                        check_type(
                            dict_td.select_one("div.rcmd_dict_dl_btn a"), Tag
                        ).get("href"),
                        str,
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
                    check_type(
                        check_type(category.select_one("a"), Tag).get("href"), str
                    )
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

    def _report_stats(self) -> None:
        log.info("Successfully downloaded/skipped %d dictionaries.", self._count)
        if not self._exceptions:
            return
        log.info("---- Detailed Exception Summary ----")
        for exception in self._exceptions:
            log.error("", exc_info=exception)

    def shutdown(self) -> None:
        """Shutdown the spider."""
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=True)
        for s in self._sessions:
            s.close()
        self._report_stats()

    def raise_for_exceptions(self) -> None:
        """
        Raise a DownloadError if any exceptions occurred during the download process.

        Raises:
            DownloadError: If any exceptions occurred during the download process.
        """
        if self._exceptions:
            msg = (
                f"{len(self._exceptions)} exceptions occurred "
                "during the download process."
            )
            raise DownloadError(msg)

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
