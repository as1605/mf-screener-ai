"""
Mutual Fund Data Provider

A production-ready utility for fetching and providing mutual fund data from Tickertape API.
Handles data fetching, caching, and provides a clean interface for data access.

Author: Claude Sonnet 4.5 Extended
Version: 1.0.0
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MfDataProviderError(Exception):
    """Base exception for MfDataProvider errors"""
    pass


class APIError(MfDataProviderError):
    """Raised when API calls fail"""
    pass


class DataNotFoundError(MfDataProviderError):
    """Raised when requested data is not found"""
    pass


class MfDataProvider:
    """
    A comprehensive data provider for mutual fund and index data.
    
    Features:
    - Fetches mutual fund listings and historical data
    - Fetches index historical data
    - Automatic caching with date-based directories
    - Retry logic for API failures
    - Rate limiting to prevent API throttling
    - Comprehensive error handling and logging
    
    Attributes:
        base_dir (str): Base directory for data storage
        data_dir (str): Current date-based data directory
        session (requests.Session): Configured session with retry logic
    """
    
    # API Configuration
    MF_LIST_URL = "https://api.tickertape.in/mf-screener/query"
    MF_CHART_URL = "https://api.tickertape.in/mutualfunds/{mfId}/charts/inter"
    INDEX_CHART_URL = "https://api.tickertape.in/stocks/charts/inter/{indexId}"
    HEADERS = {"accept-version": "8.14.0"}
    
    # Index Mappings
    INDICES = {
        "Large Cap": ".NSEI",
        "Mid Cap": ".NIMI150",
        "Small Cap": ".NISM250",
        "Total Market": ".NIFTY500",
        "Gold": "GBES"
    }
    
    # Rate limiting (for single-threaded requests)
    REQUEST_DELAY = 0.5  # seconds between requests

    # Parallel fetch settings (for fetch_all_data)
    MF_CHART_MAX_WORKERS = 12
    INDEX_CHART_MAX_WORKERS = 5

    def __init__(self, base_dir: str = './data', date: Optional[str] = None):
        """
        Initialize the MfDataProvider.
        
        Args:
            base_dir (str): Base directory for data storage. Default: './data'
            date (str, optional): Date string in 'YYYY-MM-DD' format. 
                                  If None, uses today's date.
        
        Example:
            >>> provider = MfDataProvider()  # Uses today's date
            >>> provider = MfDataProvider(date='2024-01-15')  # Uses specific date
        """
        self.base_dir = base_dir
        
        # Set data directory based on date
        if date:
            self.data_dir = os.path.join(base_dir, date)
        else:
            self.data_dir = os.path.join(base_dir, datetime.now().strftime('%Y-%m-%d'))
        
        # Create directory structure
        self._create_directories()
        
        # Configure session with retry logic
        self.session = self._create_session()
        
        logger.info(f"MfDataProvider initialized with data directory: {self.data_dir}")
    
    def _create_directories(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.data_dir, 'mf')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.data_dir, 'index')).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory structure created at {self.data_dir}")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.
        
        Returns:
            requests.Session: Configured session object
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request with error handling and rate limiting.
        
        Args:
            method (str): HTTP method ('GET' or 'POST')
            url (str): Request URL
            **kwargs: Additional arguments for requests
        
        Returns:
            dict: JSON response data
        
        Raises:
            APIError: If the request fails or returns unsuccessful response
        """
        try:
            sleep(self.REQUEST_DELAY)  # Rate limiting
            
            if method.upper() == 'POST':
                response = self.session.post(url, timeout=30, **kwargs)
            else:
                response = self.session.get(url, timeout=30, **kwargs)
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', False):
                raise APIError(f"API returned success=false for {url}")
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise APIError(f"Failed to fetch data from {url}: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid JSON response from {url}")
            raise APIError(f"Invalid JSON response from {url}") from e
    
    def fetch_all_mf_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch complete list of mutual funds with metadata.
        
        Args:
            force_refresh (bool): If True, fetch fresh data even if cached. Default: False
        
        Returns:
            pd.DataFrame: DataFrame with columns [mfId, name, aum, sector, subsector]
        
        Raises:
            APIError: If API call fails
        """
        all_file = os.path.join(self.data_dir, 'ALL.csv')
        
        # Return cached data if available and not forcing refresh
        if os.path.exists(all_file) and not force_refresh:
            logger.info("Loading cached MF list from ALL.csv")
            return pd.read_csv(all_file)
        
        logger.info("Fetching mutual fund list from API...")
        
        payload = {
            "match": {"option": ["Growth"]},
            "sortBy": "aum",
            "sortOrder": -1,
            "project": ["subsector", "option", "aum", "ret3y", "expRatio"],
            "offset": 0,
            "count": 2000,
            "mfIds": []
        }
        
        data = self._make_request('POST', self.MF_LIST_URL, headers=self.HEADERS, json=payload)
        
        # Parse response
        mf_list = []
        for mf in data.get('data', {}).get('result', []):
            mf_info = {
                'mfId': mf.get('mfId'),
                'name': mf.get('name'),
                'sector': mf.get('sector'),
            }
            
            # Extract values from the values array
            for value_obj in mf.get('values', []):
                filter_name = value_obj.get('filter')
                if filter_name == 'aum':
                    mf_info['aum'] = value_obj.get('doubleVal')
                elif filter_name == 'subsector':
                    mf_info['subsector'] = value_obj.get('strVal')
            
            mf_list.append(mf_info)
        
        # Create DataFrame
        df = pd.DataFrame(mf_list)
        
        # Save to file
        df.to_csv(all_file, index=False)
        logger.info(f"Saved {len(df)} mutual funds to {all_file}")
        
        return df
    
    def fetch_mf_chart(self, mf_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical chart data for a specific mutual fund.
        
        Args:
            mf_id (str): Mutual fund ID (e.g., 'M_PARO')
            force_refresh (bool): If True, fetch fresh data even if cached. Default: False
        
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, nav]
        
        Raises:
            APIError: If API call fails
        """
        mf_file = os.path.join(self.data_dir, 'mf', f'{mf_id}.csv')
        
        # Return cached data if available and not forcing refresh
        if os.path.exists(mf_file) and not force_refresh:
            logger.debug(f"Loading cached chart data for {mf_id}")
            return pd.read_csv(mf_file)
        
        logger.info(f"Fetching chart data for mutual fund: {mf_id}")
        
        url = self.MF_CHART_URL.format(mfId=mf_id)
        params = {'duration': 'max'}
        
        data = self._make_request('GET', url, params=params)
        
        # Parse response
        chart_data = []
        for series in data.get('data', []):
            for point in series.get('points', []):
                chart_data.append({
                    'timestamp': point.get('ts'),
                    'nav': point.get('lp')
                })
        
        if not chart_data:
            logger.warning(f"No chart data found for {mf_id}")
            return pd.DataFrame(columns=['timestamp', 'nav'])
        
        # Create DataFrame
        df = pd.DataFrame(chart_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Save to file
        df.to_csv(mf_file, index=False)
        logger.info(f"Saved {len(df)} data points for {mf_id}")
        
        return df

    def _fetch_one_mf_chart_to_disk(self, mf_id: str) -> Tuple[str, Optional[Exception]]:
        """
        Fetch a single MF chart and write to disk. Thread-safe (own session).
        Returns (mf_id, None) on success, (mf_id, exception) on failure.
        """
        mf_file = os.path.join(self.data_dir, 'mf', f'{mf_id}.csv')
        url = self.MF_CHART_URL.format(mfId=mf_id)
        params = {'duration': 'max'}
        session = self._create_session()
        try:
            response = session.get(url, params=params, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data.get('success', False):
                return (mf_id, APIError(f"API returned success=false for {mf_id}"))
            chart_data = []
            for series in data.get('data', []):
                for point in series.get('points', []):
                    chart_data.append({'timestamp': point.get('ts'), 'nav': point.get('lp')})
            if not chart_data:
                return (mf_id, None)  # write empty is ok
            df = pd.DataFrame(chart_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.to_csv(mf_file, index=False)
            return (mf_id, None)
        except Exception as e:
            return (mf_id, e)

    def _fetch_one_index_chart_to_disk(self, item: Tuple[str, str]) -> Tuple[str, Optional[Exception]]:
        """
        Fetch a single index chart and write to disk. Thread-safe (own session).
        item is (name, index_id). Returns (index_id, None) on success, (index_id, exception) on failure.
        """
        name, index_id = item
        safe_index_id = index_id.replace('.', '_')
        index_file = os.path.join(self.data_dir, 'index', f'{safe_index_id}.csv')
        url = self.INDEX_CHART_URL.format(indexId=index_id)
        params = {'duration': 'max'}
        session = self._create_session()
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data.get('success', False):
                return (index_id, APIError(f"API returned success=false for {index_id}"))
            chart_data = []
            for series in data.get('data', []):
                for point in series.get('points', []):
                    chart_data.append({'timestamp': point.get('ts'), 'nav': point.get('lp')})
            if not chart_data:
                return (index_id, None)
            df = pd.DataFrame(chart_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.to_csv(index_file, index=False)
            return (index_id, None)
        except Exception as e:
            return (index_id, e)
    
    def fetch_index_chart(self, index_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical chart data for a specific index.
        
        Args:
            index_id (str): Index ID (e.g., '.NSEI', 'GBES')
            force_refresh (bool): If True, fetch fresh data even if cached. Default: False
        
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, nav]
        
        Raises:
            APIError: If API call fails
        """
        # Sanitize index_id for filename (replace dots)
        safe_index_id = index_id.replace('.', '_')
        index_file = os.path.join(self.data_dir, 'index', f'{safe_index_id}.csv')
        
        # Return cached data if available and not forcing refresh
        if os.path.exists(index_file) and not force_refresh:
            logger.debug(f"Loading cached chart data for index {index_id}")
            return pd.read_csv(index_file)
        
        logger.info(f"Fetching chart data for index: {index_id}")
        
        url = self.INDEX_CHART_URL.format(indexId=index_id)
        params = {'duration': 'max'}
        
        data = self._make_request('GET', url, params=params)
        
        # Parse response
        chart_data = []
        for series in data.get('data', []):
            for point in series.get('points', []):
                chart_data.append({
                    'timestamp': point.get('ts'),
                    'nav': point.get('lp')
                })
        
        if not chart_data:
            logger.warning(f"No chart data found for index {index_id}")
            return pd.DataFrame(columns=['timestamp', 'nav'])
        
        # Create DataFrame
        df = pd.DataFrame(chart_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Save to file
        df.to_csv(index_file, index=False)
        logger.info(f"Saved {len(df)} data points for index {index_id}")
        
        return df
    
    def fetch_all_data(self) -> Dict[str, int]:
        """
        Fetch all data: MF list, all MF charts, and all index charts.
        Uses multithreading for MF and index charts for 5–10x speedup.

        Returns:
            dict: Counts with keys mf_success, mf_failed, index_success, index_failed.
        """
        logger.info("Starting full data fetch (parallel)...")

        # Stage 1: Fetch MF list (single request)
        df_mf = self.fetch_all_mf_list(force_refresh=True)
        mf_ids = df_mf['mfId'].tolist()
        logger.info(f"Fetched list of {len(mf_ids)} mutual funds")

        # Stage 2: Fetch all MF charts in parallel
        mf_success, mf_failed = 0, 0
        logger.info(f"Fetching MF charts with {self.MF_CHART_MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=self.MF_CHART_MAX_WORKERS) as executor:
            futures = {executor.submit(self._fetch_one_mf_chart_to_disk, mf_id): mf_id for mf_id in mf_ids}
            for idx, future in enumerate(as_completed(futures), 1):
                mf_id, err = future.result()
                if err is None:
                    mf_success += 1
                else:
                    mf_failed += 1
                    logger.error(f"Failed to fetch chart for {mf_id}: {err}")
                if idx % 50 == 0:
                    logger.info(f"MF charts progress: {idx}/{len(mf_ids)}")
        logger.info(f"MF charts done: {mf_success} ok, {mf_failed} failed")

        # Stage 3: Fetch all index charts in parallel
        index_items = list(self.INDICES.items())
        index_success, index_failed = 0, 0
        logger.info(f"Fetching index charts with {self.INDEX_CHART_MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=self.INDEX_CHART_MAX_WORKERS) as executor:
            futures = {executor.submit(self._fetch_one_index_chart_to_disk, item): item for item in index_items}
            for future in as_completed(futures):
                index_id, err = future.result()
                if err is None:
                    index_success += 1
                else:
                    index_failed += 1
                    logger.error(f"Failed to fetch index {index_id}: {err}")
        logger.info(f"Index charts done: {index_success} ok, {index_failed} failed")
        logger.info("Full data fetch completed!")

        return {
            "mf_success": mf_success,
            "mf_failed": mf_failed,
            "index_success": index_success,
            "index_failed": index_failed,
        }
    
    # ==================== Public API Methods ====================
    
    def list_all_mf(self) -> pd.DataFrame:
        """
        Get list of all mutual funds with metadata.
        
        Returns:
            pd.DataFrame: DataFrame with columns [mfId, name, aum, sector, subsector]
        
        Example:
            >>> provider = MfDataProvider()
            >>> df = provider.list_all_mf()
            >>> print(df.head())
        """
        try:
            return self.fetch_all_mf_list(force_refresh=False)
        except Exception as e:
            logger.error(f"Error fetching MF list: {str(e)}")
            raise
    
    def list_mf_by_sector(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get mutual funds organized by sector and subsector.
        
        Returns:
            dict: Nested dictionary with structure:
                  {"sector": {"subsector": ["mfId1", "mfId2", ...]}}
        
        Example:
            >>> provider = MfDataProvider()
            >>> sectors = provider.list_mf_by_sector()
            >>> print(sectors['Equity']['Flexi Cap Fund'])
            ['M_PARO', 'M_AXIS', ...]
        """
        df = self.list_all_mf()
        
        result = {}
        for sector in df['sector'].unique():
            if pd.isna(sector):
                continue
            
            result[sector] = {}
            sector_df = df[df['sector'] == sector]
            
            for subsector in sector_df['subsector'].unique():
                if pd.isna(subsector):
                    continue
                
                mf_ids = sector_df[sector_df['subsector'] == subsector]['mfId'].tolist()
                result[sector][subsector] = mf_ids
        
        return result
    
    def get_mf_chart(self, mf_id: str) -> pd.DataFrame:
        """
        Get historical chart data for a specific mutual fund.
        
        Args:
            mf_id (str): Mutual fund ID (e.g., 'M_PARO')
        
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, nav]
        
        Raises:
            DataNotFoundError: If the mutual fund ID is invalid
        
        Example:
            >>> provider = MfDataProvider()
            >>> df = provider.get_mf_chart('M_PARO')
            >>> print(df.tail())
        """
        try:
            return self.fetch_mf_chart(mf_id, force_refresh=False)
        except APIError as e:
            raise DataNotFoundError(f"Failed to get chart for {mf_id}: {str(e)}") from e
    
    def list_indices(self) -> Dict[str, str]:
        """
        Get list of available indices.
        
        Returns:
            dict: Dictionary mapping index names to their IDs
                  {"name": "indexId"}
        
        Example:
            >>> provider = MfDataProvider()
            >>> indices = provider.list_indices()
            >>> print(indices)
            {'Large Cap': '.NSEI', 'Mid Cap': '.NIMI150', ...}
        """
        return self.INDICES.copy()
    
    def get_index_chart(self, index_id: str) -> pd.DataFrame:
        """
        Get historical chart data for a specific index.
        
        Args:
            index_id (str): Index ID (e.g., '.NSEI', 'GBES')
                           Can also use index name (e.g., 'Large Cap')
        
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, nav]
        
        Raises:
            DataNotFoundError: If the index ID is invalid
        
        Example:
            >>> provider = MfDataProvider()
            >>> df = provider.get_index_chart('.NSEI')
            >>> # Or use name
            >>> df = provider.get_index_chart('Large Cap')
            >>> print(df.tail())
        """
        # Check if index_id is actually a name
        if index_id in self.INDICES:
            index_id = self.INDICES[index_id]
        
        try:
            return self.fetch_index_chart(index_id, force_refresh=False)
        except APIError as e:
            raise DataNotFoundError(f"Failed to get chart for index {index_id}: {str(e)}") from e


def main() -> None:
    """
    Entry point: fetch all MFs and indices using parallel data fetch, then print summary.
    Called from __main__.
    """
    logger.info("Starting MfDataProvider - Fetching all data...")
    try:
        provider = MfDataProvider()
        print("\n" + "=" * 70)
        print("FETCHING ALL MUTUAL FUNDS AND INDICES (parallel)")
        print("=" * 70)
        print("\nStages: 1) MF list  2) MF charts (threaded)  3) Index charts (threaded)")
        print("=" * 70)

        counts = provider.fetch_all_data()

        print("\n" + "=" * 70)
        print("FETCH SUMMARY")
        print("=" * 70)
        print(f"\nData directory: {provider.data_dir}")
        print(f"\nMutual Funds: {counts['mf_success']} charts ok, {counts['mf_failed']} failed")
        print(f"Indices: {counts['index_success']} ok, {counts['index_failed']} failed")
        if counts['mf_failed'] + counts['index_failed'] == 0:
            print("\nAll data fetched successfully.")
        else:
            print("\nCompleted with some errors (see logs).")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n\nFetch interrupted by user (Ctrl+C). Partial data may have been saved.")
        logger.warning("Data fetch interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error during data fetch: {str(e)}", exc_info=True)
        print(f"\nFatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
