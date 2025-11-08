import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SeasonInfo:
    season_id: str
    label: str
    regular_path: str  # Absolute path to regular-season overview page


class PremierLeagueScraper:
    """
    Scraper for Premier League match data.

    NOTE: This scraper currently uses SHL (Swedish Hockey League) data source as a template.
    TODO: Update to use Premier League data sources (e.g., football-data.co.uk, Premier League API, etc.)

    Returns a combined DataFrame containing completed results (with FTHG/FTAG)
    and upcoming fixtures (with FTHG/FTAG set to NaN).
    """

    # TODO: Update these URLs to Premier League data sources
    BASE_URL = "https://stats.swehockey.se"  # TEMPORARY - needs Premier League source
    DEFAULT_SEASON_ID = "18263"  # TEMPORARY - needs Premier League season ID

    DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
    FULL_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")
    SCORE_PATTERN = re.compile(r"^\d+\s*-\s*\d+$")
    # TODO: Update keywords for Premier League data sources
    REGULAR_LABEL_KEYWORDS = ("premier league", "epl", "regular", "league")  # Updated for Premier League
    EXCLUDED_LABEL_KEYWORDS = ("playoff", "cup", "final", "champions", "europa")

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 30):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
            }
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.season_map = self._load_season_map()

    def scrape_matches(self, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Fetch historical Premier League match results and upcoming fixtures.

        Args:
            seasons: Iterable of season start years (e.g. 2023 for 2023-24). When None,
                     defaults to the three most recent seasons available.

        Returns:
            DataFrame containing Date, HomeTeam, AwayTeam, FTHG, FTAG, Season, SeasonStart.
        """
        if seasons is None:
            seasons = self._default_seasons()
        elif isinstance(seasons, int):
            seasons = [seasons]

        all_frames: List[pd.DataFrame] = []

        for season_start in seasons:
            info = self.season_map.get(season_start)
            if not info:
                self.logger.warning("Season %s not found in Premier League schedule options", season_start)
                continue

            try:
                results_df, fixtures_df = self._fetch_season_data(info, season_start)
            except Exception as exc:
                self.logger.error("Failed to fetch season %s (%s): %s", season_start, info.label, exc)
                continue

            season_frames = [df for df in (results_df, fixtures_df) if not df.empty]
            if season_frames:
                all_frames.append(pd.concat(season_frames, ignore_index=True))

        if not all_frames:
            self.logger.warning("No Premier League data collected for requested seasons")
            return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"])

        combined = pd.concat(all_frames, ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined = combined.dropna(subset=["Date"]).reset_index(drop=True)
        combined["Match"] = combined["HomeTeam"] + " - " + combined["AwayTeam"]
        self.logger.info("Collected %d rows across %d seasons", len(combined), len(all_frames))
        return combined

    def get_upcoming_fixtures(self, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Retrieve upcoming fixtures (no final score recorded yet).

        Args:
            seasons: Optional iterable of season start years. Defaults to current season.

        Returns:
            DataFrame with Date, HomeTeam, AwayTeam columns for fixtures in the future.
        """
        if seasons is None:
            current_year = datetime.now().year
            seasons = [current_year]
        fixtures_df = self.scrape_matches(seasons)
        fixtures_df = fixtures_df[fixtures_df["FTHG"].isna()].copy()
        if fixtures_df.empty:
            return fixtures_df[["Date", "HomeTeam", "AwayTeam"]]

        today = pd.Timestamp.now().normalize()
        fixtures_df = fixtures_df[fixtures_df["Date"] >= today]
        return fixtures_df[["Date", "HomeTeam", "AwayTeam", "Season", "SeasonStart", "Match"]]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _default_seasons(self) -> List[int]:
        sorted_years = sorted(self.season_map.keys(), reverse=True)
        return sorted_years[:3]

    def _load_season_map(self) -> Dict[int, SeasonInfo]:
        url = f"{self.BASE_URL}/ScheduleAndResults/Overview/{self.DEFAULT_SEASON_ID}"
        html = self._fetch_html(url)
        if not html:
            self.logger.warning("Season dropdown unavailable; defaulting to current season only")
            current_year = datetime.now().year
            path = f"/ScheduleAndResults/Overview/{self.DEFAULT_SEASON_ID}"
            return {current_year: SeasonInfo(self.DEFAULT_SEASON_ID, f"{current_year}-{current_year+1}", path)}

        soup = BeautifulSoup(html, "html.parser")
        season_options = soup.select('select[onchange="doNavigate(this)"] option')

        season_map: Dict[int, SeasonInfo] = {}
        for option in season_options:
            label = option.text.strip()
            value = (option.get("value") or option.get("id") or "").strip()
            match = re.search(r"/ScheduleAndResults/(Overview|Schedule)/(\d+)", value)
            year_match = re.search(r"(\d{4})", label)

            if not match or not year_match:
                continue

            season_id = match.group(2)
            start_year = int(year_match.group(1))
            regular_path = self._resolve_regular_path(value)
            season_map[start_year] = SeasonInfo(season_id=season_id, label=label, regular_path=regular_path)

        if season_map:
            return season_map

        self.logger.warning("Failed to parse season dropdown options; using fallback season id")
        current_year = datetime.now().year
        path = f"/ScheduleAndResults/Overview/{self.DEFAULT_SEASON_ID}"
        return {current_year: SeasonInfo(self.DEFAULT_SEASON_ID, f"{current_year}-{current_year+1}", path)}

    def _resolve_regular_path(self, base_path: str) -> str:
        """
        Identify the specific statistics path representing the regular season.
        Falls back to the provided base path when no dedicated option exists.
        """
        base_html = self._fetch_html(f"{self.BASE_URL}{base_path}")
        if not base_html:
            return base_path

        soup = BeautifulSoup(base_html, "html.parser")
        selects = soup.find_all("select")
        if len(selects) < 2:
            return base_path

        stage_select = selects[1]
        candidates = []

        for opt in stage_select.find_all("option"):
            value = opt.get("value")
            if not value or value in {"0", "-1"}:
                continue
            text = opt.get_text(strip=True)
            normalized = text.lower()
            if any(excl in normalized for excl in self.EXCLUDED_LABEL_KEYWORDS):
                continue
            if any(key in normalized for key in self.REGULAR_LABEL_KEYWORDS):
                return value
            candidates.append(value)

        # Fallback to first candidate (often the base competition) if no keyword matched
        return candidates[0] if candidates else base_path

    def _fetch_season_data(self, info: SeasonInfo, season_start: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path_type, regular_id = self._parse_path(info.regular_path)

        schedule_path = info.regular_path if path_type == "Schedule" else f"/ScheduleAndResults/Schedule/{regular_id}"
        schedule_html = self._fetch_html(f"{self.BASE_URL}{schedule_path}")
        if not schedule_html:
            raise RuntimeError(f"Unable to load schedule page {schedule_path}")

        rows = self._extract_rows(schedule_html)
        results_df = self._rows_to_results(rows, info.label, season_start)
        fixtures_df = self._rows_to_fixtures(rows, info.label, season_start)
        return results_df, fixtures_df

    def _parse_path(self, path: str) -> Tuple[str, str]:
        match = re.search(r"/ScheduleAndResults/(Overview|Schedule)/(\d+)", path)
        if not match:
            raise ValueError(f"Unrecognized season path: {path}")
        return match.group(1), match.group(2)

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            self.logger.debug("HTTP error for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _rows_to_results(self, rows: List[Dict[str, Optional[str]]], label: str, season_start: int) -> pd.DataFrame:
        results = []
        for row in rows:
            if row["score"] is None:
                continue
            try:
                date_obj = datetime.strptime(row["date"], "%Y-%m-%d")
            except ValueError:
                continue

            home_goals, away_goals = row["score"]
            results.append(
                {
                    "Date": date_obj,
                    "HomeTeam": row["home_team"],
                    "AwayTeam": row["away_team"],
                    "FTHG": home_goals,
                    "FTAG": away_goals,
                    "Season": label,
                    "SeasonStart": season_start,
                    "ScoreDetail": row.get("detail"),
                }
            )
        return pd.DataFrame(results)

    def _rows_to_fixtures(self, rows: List[Dict[str, Optional[str]]], label: str, season_start: int) -> pd.DataFrame:
        fixtures = []
        for row in rows:
            if row["score"] is not None:
                continue
            try:
                date_obj = datetime.strptime(row["date"], "%Y-%m-%d")
            except ValueError:
                continue

            fixtures.append(
                {
                    "Date": date_obj,
                    "HomeTeam": row["home_team"],
                    "AwayTeam": row["away_team"],
                    "FTHG": None,
                    "FTAG": None,
                    "Season": label,
                    "SeasonStart": season_start,
                    "ScoreDetail": None,
                }
            )
        return pd.DataFrame(fixtures)

    def _extract_rows(self, html: str) -> List[Dict[str, Optional[str]]]:
        soup = BeautifulSoup(html, "html.parser")
        table = self._locate_results_table(soup)
        if not table:
            return []

        rows: List[Dict[str, Optional[str]]] = []
        current_date: Optional[str] = None

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            row_data, current_date = self._parse_row(cells, current_date)
            if row_data:
                rows.append(row_data)

        return rows

    def _locate_results_table(self, soup: BeautifulSoup):
        # Try to find a table linked to a heading containing "Result"
        for heading in soup.find_all(["h2", "th"]):
            text = heading.get_text(strip=True).lower()
            if "result" in text and heading.name == "h2":
                table = heading.find_parent("table")
                if table:
                    content = table.find("table", class_="tblContent")
                    if content:
                        return content
            if heading.name == "th" and "result" in text:
                return heading.find_parent("table")
        # Fallback: first table containing a "Result" header cell
        for table in soup.find_all("table"):
            if table.find("th", string=lambda s: s and "Result" in s):
                return table
        return None

    def _parse_row(self, cells, current_date: Optional[str]) -> Tuple[Optional[Dict[str, Optional[str]]], Optional[str]]:
        texts = [self._normalize_text(cell.get_text(" ", strip=True)) for cell in cells]

        # Update current date if this row provides one
        new_date = current_date
        for text in texts:
            if self.FULL_DATE_PATTERN.match(text):
                new_date = text
                break

        if not new_date:
            return None, current_date

        teams_text = next((text for text in texts if " - " in text), None)
        if not teams_text:
            return None, new_date

        home_team, away_team = [part.strip() for part in teams_text.split(" - ", 1)]
        if not home_team or not away_team:
            return None, new_date

        score_text = next((text for text in texts if self.SCORE_PATTERN.match(text)), None)
        score: Optional[Tuple[int, int]] = None
        if score_text:
            try:
                home_goals, away_goals = [int(part.strip()) for part in score_text.split("-")]
                score = (home_goals, away_goals)
            except ValueError:
                score = None

        detail_text = next((text for text in texts if "(" in text and ")" in text), None)

        return {
            "date": new_date,
            "home_team": home_team,
            "away_team": away_team,
            "score": score,
            "detail": detail_text,
        }, new_date

    @staticmethod
    def _normalize_text(value: str) -> str:
        # Normalize non-breaking spaces and en/em dashes to regular spaces/hyphen
        return (
            value.replace("\u00a0", " ")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .strip()
        )
