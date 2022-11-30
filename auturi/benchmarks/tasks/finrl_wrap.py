import itertools
import pathlib
import pickle

import pandas as pd
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRADE_END_DATE, TRAIN_START_DATE
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from stable_baselines3.common.logger import configure

SAVE_ASSET_PATH = pathlib.Path(__file__).parent.resolve() / "finrl_assets"

TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2020-07-01"
TRADE_START_DATE = "2020-07-01"
TRADE_END_DATE = "2021-10-31"


def _get_yahoo_df():
    yahoo_path = SAVE_ASSET_PATH / "finrl_yahoo.csv"
    if yahoo_path.exists():
        df = pd.read_csv(yahoo_path)
    else:
        df = YahooDownloader(
            start_date=TRAIN_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=config_tickers.DOW_30_TICKER,
        ).fetch_data()

        df.to_csv(yahoo_path)

    return df


def _get_feature_engineer():
    fe_path = SAVE_ASSET_PATH / "finrl_feature.pkl"
    if fe_path.exists():
        with open(fe_path, "rb") as f:
            fe = pickle.load(f)

    else:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False,
        )

        with open(fe_path, "wb") as f:
            pickle.dump(fe, f)

    return fe


def get_env():
    env_path = SAVE_ASSET_PATH / "finrl_env.pkl"
    if env_path.exists():
        with open(env_path, "rb") as f:
            env = pickle.load(f)
            return env

    else:
        df = _get_yahoo_df()
        fe = _get_feature_engineer()

        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(
            pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
        )
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
            processed, on=["date", "tic"], how="left"
        )
        processed_full = processed_full[processed_full["date"].isin(processed["date"])]
        processed_full = processed_full.sort_values(["date", "tic"])

        processed_full = processed_full.fillna(0)
        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
        stock_dimension = len(train.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
        }

        env = StockTradingEnv(df=train, **env_kwargs).get_sb_env()
        with open(env_path, "wb") as f:
            pickle.dump(env, f)

        return env