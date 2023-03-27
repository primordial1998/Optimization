import copy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pypfopt.plotting import plot_efficient_frontier

import yfinance as yf

# ----------------- Plotting function for efficient frontier ----------------- #


def custom_efficient_frontier(ef, risk_free_rate, show_assets=True, figsize=(12, 10)):

    # Create a copy of the efficient frontier class instance
    ef_copy = copy.deepcopy(ef)

    fig, ax = plt.subplots(figsize=figsize)

    # Create copies for each portfolio
    ef_max_sharpe = copy.deepcopy(ef_copy)
    ef_min_vol = copy.deepcopy(ef_copy)
    ef_max_quadratic_utility = copy.deepcopy(ef_copy)

    # Plot the efficient frontier
    plot_efficient_frontier(ef_copy, ax=ax, show_assets=show_assets)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*",
               s=200, c="r", label="Max Sharpe")

    # Find the minimum volatility portfolio
    ef_min_vol.min_volatility()
    ret_min_vol, std_min_vol, _ = ef_min_vol.portfolio_performance()
    ax.scatter(std_min_vol, ret_min_vol, marker="p",
               s=200, c="g", label="Min Volatility")

    # Find max quadratic utility portfolio
    ef_max_quadratic_utility.max_quadratic_utility(risk_aversion=0.1)
    ret_max_quadratic_utility, std_max_quadratic_utility, _ = ef_max_quadratic_utility.portfolio_performance()
    ax.scatter(std_max_quadratic_utility, ret_max_quadratic_utility,
               marker="^", s=200, c="b", label="Max Quadratic Utility")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef_copy.n_assets), n_samples)
    # Compute the returns for each random portfolio
    rets = w.dot(ef_copy.expected_returns)
    # Compute the standard deviation of each random portfolio
    stds = np.sqrt(np.diag(w @ ef_copy.cov_matrix @ w.T))
    sharpes = rets / stds  # Compute the Sharpe ratio for each random portfolio
    # Plot the random portfolios, using a colormap to indicate the Sharpe ratio
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.tight_layout()

    return ax

# --------------- A Class for easy company long name retrieval --------------- #


def func(t: Union[List[str], Tuple[str]]):
    pass


class Nasdaq100:

    def __init__(self, tickers: Union[List[str], Tuple[str]]):
        self.tickers = tickers
        self.mapping = {
            'AAPL': 'Apple Inc.',
            'ABBV': 'AbbVie Inc.',
            'ABMD': 'Abiomed Inc.',
            'ACN': 'Accenture plc',
            'ADBE': 'Adobe Inc.',
            'ADI': 'Analog Devices Inc.',
            'ADP': 'Automatic Data Processing Inc.',
            'ADSK': 'Autodesk Inc.',
            'AEP': 'American Electric Power Company Inc.',
            'ALGN': 'Align Technology Inc.',
            'AMAT': 'Applied Materials Inc.',
            'AMD': 'Advanced Micro Devices Inc.',
            'AMGN': 'Amgen Inc.',
            'AMZN': 'Amazon.com Inc.',
            'ANSS': 'ANSYS Inc.',
            'ANTM': 'Anthem Inc.',
            'ASML': 'ASML Holding NV',
            'ATVI': 'Activision Blizzard Inc.',
            'AVGO': 'Broadcom Inc.',
            'BIDU': 'Baidu Inc.',
            'BIIB': 'Biogen Inc.',
            'BKNG': 'Booking Holdings Inc.',
            'BMRN': 'BioMarin Pharmaceutical Inc.',
            'BMY': 'Bristol-Myers Squibb Company',
            'CDNS': 'Cadence Design Systems Inc.',
            'CDW': 'CDW Corp.',
            'CERN': 'Cerner Corp.',
            'CHKP': 'Check Point Software Technologies Ltd.',
            'CHTR': 'Charter Communications Inc.',
            'CMCSA': 'Comcast Corp.',
            'CME': 'CME Group Inc.',
            'COST': 'Costco Wholesale Corp.',
            'CSCO': 'Cisco Systems Inc.',
            'CSX': 'CSX Corp.',
            'CTAS': 'Cintas Corp.',
            'CTSH': 'Cognizant Technology Solutions Corp.',
            'CTXS': 'Citrix Systems Inc.',
            'DISCA': 'Discovery Inc. Class A',
            'DISCK': 'Discovery Inc. Class C',
            'DISH': 'DISH Network Corp.',
            'DLTR': 'Dollar Tree Inc.',
            'DOCU': 'DocuSign Inc.',
            'DOGE-USD': 'Dogecoin',
            'DXCM': 'DexCom Inc.',
            'EA': 'Electronic Arts Inc.',
            'EBAY': 'eBay Inc.',
            'EXC': 'Exelon Corp.',
            'EXPE': 'Expedia Group Inc.',
            'FAST': 'Fastenal Co.',
            'FISV': 'Fiserv Inc.',
            'FOX': 'Fox Corporation Class B',
            'FOXA': 'Fox Corporation Class A',
            'GILD': 'Gilead Sciences Inc.',
            'GOOG': 'Alphabet Inc. Class C',
            'GOOGL': 'Alphabet Inc. Class A',
            'GRMN': 'Garmin Ltd.',
            'GILD': 'Gilead Sciences Inc.',
            'HD': 'Home Depot Inc.',
            'HON': 'Honeywell International Inc.',
            'IDXX': 'IDEXX Laboratories Inc.',
            'ILMN': 'Illumina Inc.',
            'INCY': 'Incyte Corp.',
            'INTC': 'Intel Corp.',
            'INTU': 'Intuit Inc.',
            'ISRG': 'Intuitive Surgical Inc.',
            'JD': 'JD.com Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'KDP': 'Keurig Dr Pepper Inc.',
            'KHC': 'Kraft Heinz Co.',
            'KLAC': 'KLA Corp.',
            'LBTYA': 'Liberty Global plc Class A',
            'LBTYK': 'Liberty Global plc Class C',
            'LRCX': 'Lam Research Corp.',
            'LULU': 'Lululemon Athletica Inc.',
            'MAR': 'Marriott International Inc.',
            "MAR": "Marriott International Inc",
            "MCHP": "Microchip Technology Inc",
            "MDLZ": "Mondelez International Inc",
            "MELI": "MercadoLibre Inc",
            'META': 'Meta Platforms Inc.',
            "MNST": "Monster Beverage Corp",
            "MSFT": "Microsoft Corp",
            "MU": "Micron Technology Inc",
            "MXIM": "Maxim Integrated Products Inc",
            "NFLX": "Netflix Inc",
            "NTES": "NetEase Inc",
            "NVDA": "NVIDIA Corp",
            "NXPI": "NXP Semiconductors NV",
            "ORLY": "O'Reilly Automotive Inc",
            "PAYX": "Paychex Inc",
            "PCAR": "PACCAR Inc",
            "PDD": "Pinduoduo Inc",
            "PEP": "PepsiCo Inc",
            "PFE": "Pfizer Inc",
            "PKG": "Packaging Corp of America",
            "PYPL": "PayPal Holdings Inc",
            "QCOM": "QUALCOMM Inc",
            "REGN": "Regeneron Pharmaceuticals Inc",
            "ROST": "Ross Stores Inc",
            "SBUX": "Starbucks Corp",
            "SGEN": "Seagen Inc",
            "SIRI": "Sirius XM Holdings Inc",
            "SNPS": "Synopsys Inc",
            "SPLK": "Splunk Inc",
            "SWKS": "Skyworks Solutions Inc",
            "TCOM": "Trip.com Group Ltd",
            "TEAM": "Atlassian Corporation Plc",
            "TSLA": "Tesla Inc",
            "TXN": "Texas Instruments Inc",
            "VRSK": "Verisk Analytics Inc",
            "VRTX": "Vertex Pharmaceuticals Inc",
            "WBA": "Walgreens Boots Alliance Inc",
            "WDAY": "Workday Inc",
            "XEL": "Xcel Energy Inc",
            "XLNX": "Xilinx Inc",
            "ZM": "Zoom Video Communications Inc"
        }
        self.industries = {
            'AAPL': 'Technology',
            'ABBV': 'Healthcare',
            'ABMD': 'Healthcare',
            'ACN': 'Technology',
            'ADBE': 'Technology',
            'ADI': 'Technology',
            'ADP': 'Business Services',
            'ADSK': 'Technology',
            'AEP': 'Utilities',
            'ALGN': 'Healthcare',
            'AMAT': 'Technology',
            'AMD': 'Technology',
            'AMGN': 'Healthcare',
            'AMZN': 'Retail',
            'ANSS': 'Technology',
            'ANTM': 'Healthcare',
            'ASML': 'Technology',
            'ATVI': 'Technology',
            'AVGO': 'Technology',
            'BIDU': 'Technology',
            'BIIB': 'Healthcare',
            'BKNG': 'Travel & Hospitality',
            'BMRN': 'Healthcare',
            'BMY': 'Healthcare',
            'CDNS': 'Technology',
            'CDW': 'Business Services',
            'CERN': 'Healthcare',
            'CHKP': 'Technology',
            'CHTR': 'Telecom',
            'CMCSA': 'Telecom',
            'CME': 'Financial Services',
            'COST': 'Retail',
            'CSCO': 'Technology',
            'CSX': 'Transportation',
            'CTAS': 'Business Services',
            'CTSH': 'Technology',
            'CTXS': 'Technology',
            'DISCA': 'Media & Entertainment',
            'DISCK': 'Media & Entertainment',
            'DISH': 'Telecom',
            'DLTR': 'Retail',
            'DOCU': 'Technology',
            'DOGE-USD': 'Cryptocurrency',
            'DXCM': 'Healthcare',
            'EA': 'Technology',
            'EBAY': 'Retail',
            'EXC': 'Utilities',
            'EXPE': 'Travel & Hospitality',
            'FAST': 'Retail',
            'FISV': 'Business Services',
            'FOX': 'Media & Entertainment',
            'FOXA': 'Media & Entertainment',
            'GILD': 'Healthcare',
            'GOOG': 'Technology',
            'GOOGL': 'Technology',
            'GRMN': 'Technology',
            'HD': 'Retail',
            'HON': 'Manufacturing',
            'IDXX': 'Healthcare',
            'ILMN': 'Healthcare',
            'INCY': 'Healthcare',
            'INTC': 'Technology',
            'INTU': 'Technology',
            'ISRG': 'Healthcare',
            'JD': 'Retail',
            'JPM': 'Financial Services',
            'KDP': 'Food & Beverage',
            'KHC': 'Food & Beverage',
            'KLAC': 'Technology',
            'LBTYA': 'Telecom',
            'LBTYK': 'Telecom',
            'LRCX': 'Technology',
            'LULU': 'Retail',
            'MAR': 'Travel & Hospitality',
            'MCHP': 'Technology',
            'MDLZ': 'Food & Beverage',
            'MELI': 'Retail',
            'META': 'Technology',
            'MNST': 'Food & Beverage',
            'MSFT': 'Technology',
            'MU': 'Technology',
            'MXIM': 'Technology',
            'NFLX': 'Media & Entertainment',
            'NTES': 'Technology',
            'NVDA': 'Technology',
            'NXPI': 'Technology',
            "ORLY": "Consumer Discretionary",
            "PAYX": "Technology",
            "PCAR": "Industrials",
            "PDD": "Consumer Discretionary",
            "PEP": "Consumer Staples",
            "PFE": "Healthcare",
            "PKG": "Materials",
            "PYPL": "Technology",
            "QCOM": "Technology",
            "REGN": "Healthcare",
            "ROST": "Consumer Discretionary",
            "SBUX": "Consumer Discretionary",
            "SGEN": "Healthcare",
            "SIRI": "Communication Services",
            "SNPS": "Technology",
            "SPLK": "Technology",
            "SWKS": "Technology",
            "TCOM": "Consumer Discretionary",
            "TEAM": "Technology",
            "TSLA": "Consumer Discretionary",
            "TXN": "Technology",
            "VRSK": "Industrials",
            "VRTX": "Healthcare",
            "WBA": "Healthcare",
            "WDAY": "Technology",
            "XEL": "Utilities",
            "XLNX": "Technology",
            "ZM": "Technology"
        }

    # Retrieve company long name from ticker
    @property
    def tickers(self):
        return self._tickers

    # Validate ticker input
    @tickers.setter
    def tickers(self, tickers):
        if not isinstance(tickers, (list, tuple)) and all(isinstance(ticker, str) for ticker in tickers):
            raise TypeError("tickers must be a list or tuple of strings")
        self._tickers = tickers

    def get_company_names(self) -> List[str]:

        if not all(ticker in self.mapping for ticker in self.tickers):
            not_found = [
                ticker for ticker in self.tickers if ticker not in self.mapping]
            raise KeyError(
                f"Invalid ticker: {not_found}. Please use a valid ticker from the Nasdaq 100 index.")

        company_names = [self.mapping[key] for key in self.tickers]

        return company_names

    def get_industries(self) -> List[str]:
        if not all(ticker in self.mapping for ticker in self.tickers):
            not_found = [
                ticker for ticker in self.tickers if ticker not in self.mapping]
            raise KeyError(
                f"Invalid ticker: {not_found}. Please use a valid ticker from the Nasdaq 100 index.")

        industries = [self.industries[key] for key in self.tickers]

        return industries

    def get_prices(self, date: str) -> List[float]:
        if not all(ticker in self.mapping for ticker in self.tickers):
            not_found = [
                ticker for ticker in self.tickers if ticker not in self.mapping]
            raise KeyError(
                f"Invalid ticker: {not_found}. Please use a valid ticker from the Nasdaq 100 index.")

        # End date is the next day to ensure that the last day is included
        end_date = (pd.to_datetime(date) +
                    pd.to_timedelta(1, unit='d')).strftime('%Y-%m-%d')

        prices = yf.download(
            tickers=' '.join(self.tickers),
            start=date,
            end=end_date,
            show_errors=False
        )['Open']

        if prices.shape[0] == 0:
            raise ValueError(f"Invalid date: {date}. No data for given date.")

        return prices
