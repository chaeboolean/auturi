import itertools
import pathlib
import pickle

import pandas as pd
from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

SAVE_ASSET_PATH = pathlib.Path(__file__).parent.resolve() / "finrl_assets"


def get_stock_env():
    env_path = SAVE_ASSET_PATH / "finrl_stock.pkl"
    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2020-07-01"
    TRADE_START_DATE = "2020-07-01"
    TRADE_END_DATE = "2021-10-31"

    if env_path.exists():
        with open(env_path, "rb") as f:
            env = pickle.load(f)
            return env

    df = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

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

    env = StockTradingEnv(df=train, **env_kwargs).get_sb_env()[0]
    with open(env_path, "wb") as f:
        pickle.dump(env, f)

    return env


def get_portfolio_env():
    env_path = SAVE_ASSET_PATH / "finrl_portfolio.pkl"

    if env_path.exists():
        with open(env_path, "rb") as f:
            env = pickle.load(f)
            return env

    df = YahooDownloader(
        start_date="2008-01-01",
        end_date="2021-09-02",
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True, use_turbulence=False, user_defined_feature=False
    )

    df = fe.preprocess_data(df)
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame(
        {
            "date": df.date.unique()[lookback:],
            "cov_list": cov_list,
            "return_list": return_list,
        }
    )
    df = df.merge(df_cov, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    train = data_split(df, "2009-01-01", "2020-06-30")
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    tech_indicator_list = ["macd", "rsi_30", "cci_30", "dx_30"]
    feature_dimension = len(tech_indicator_list)
    print(f"Feature Dimension: {feature_dimension}")
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-1,
    }

    env = StockPortfolioEnv(df=train, **env_kwargs).get_sb_env()[0]
    with open(env_path, "wb") as f:
        pickle.dump(env, f)

    return env


def make_finrl_env(task_id):
    if task_id == "stock":
        return get_stock_env()
    elif task_id == "portfolio":
        return get_portfolio_env()


env = get_portfolio_env()

print(env.observation_space.shape)
print(env.reset().shape)
print(env.action_space.shape)
action = env.action_space.sample()
print(env.step([action]))
