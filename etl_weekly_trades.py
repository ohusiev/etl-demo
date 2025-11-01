
import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt

# ---------- EXTRACT ----------
def extract(input_path: str) -> pd.DataFrame:
    """
    Extract step: reads trades CSV.
    Expected columns: timestamp, client_type, user_id, symbol, side, quantity, price
    """
    df_trades = pd.read_csv(input_path, delimiter=",")
    print(df_trades.columns)
    # Check the columns and ensure timestamp is datetime
    df_trades = input_diagnostics(df_trades)
    return df_trades

def input_diagnostics(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a timestamp to the Monday of its week (week_start_date).
    """
    print ("Перевірка коректності timestamp та пустих значень інших полів:\n")
    print("Кількість рядків де хоча б один з атрибутів NaT:", df_trades.isna().any(axis =1).sum())
    print ("Cкринінг cathedorical data types:")
    cols_category = ["client_type", "symbol", "side"]
    for c in cols_category:
        vals = df_trades[c].unique()
        print(f" \n{c} ({len(vals)}):\n{vals}")
    
    print ("\n Перевести  'quantity','price' в to_numeric data types")
    cols_numeric = ["quantity","price"]
    for c in cols_numeric:
        df_trades[c] = pd.to_numeric(df_trades[c], errors="coerce")

    # Timestamp Check
    ts_tmp=pd.to_datetime(df_trades['timestamp'], format="%Y-%m-%d %H:%M:%S", errors='coerce') # to set all invalid timestamp parsing to NaN
    print ("Кількість рядків з некоректним timestamp:", ts_tmp.isna().sum())
    df_trades['timestamp_to_datetime'] = ts_tmp # made to create a temporary column for diagnostincs what exact where in invalid timestamps 
    print ("Некоректні записи в timestamp:\n",df_trades.loc[df_trades['timestamp_to_datetime'].isna(), "timestamp"].unique())
    df_trades['timestamp'] = ts_tmp # Replace original timestamp with parsed datetime
    df_trades = df_trades.drop(columns=['timestamp_to_datetime'])

    # Keep only rows without missing values (avoids FutureWarning about any on datetime64)
    mask = df_trades.notna().all(axis=1)
    df_trades = df_trades[mask]
    print("\nДані очищено від рядків де хочаб один з атрибутів NaT. Залишилось рядків:", len(df_trades))
    return df_trades

# ---------- TRANSFORM ----------
def transform(df_trades: pd.DataFrame, compute_pnl: bool = False) -> pd.DataFrame:
    """
    Transform step:
      - Adds week_start_date (Monday)
      - Aggregates by [week_start_date, client_type, user_id, symbol]
      - Calculates total_volume (sum of quantity), trade_count
      - Optionally calculates total_pnl if compute_pnl=True and sufficient inputs provided.

    PnL note: This template treats PnL as optional. For test assignement a logic for pnl calculation:
        - The average buy price defines your cost basis.
        - Realized PnL measures gains on quantities already sold.
        - Unrealized PnL values remaining inventory at the latest market price.
        - Total PnL = Realized + Unrealized → the trader’s complete profit or loss as of now
    """
    # To keep timezone aware timestamp (Вибрав таке рішення, хоча видається не важливим для семплу з прикладу, але в контексті реальних даних, певно що так)
    df_trades['week_start_date']= (df_trades.timestamp.dt.normalize() - pd.to_timedelta(df_trades.timestamp.dt.weekday, unit="D"))
    df_trades['week_start_date'] = df_trades['week_start_date'].dt.date

    # Calculate total_volume and aggregate
    df_trades["total_volume"] = df_trades["quantity"].mul(df_trades["price"])
    df_trades['buy_qty']   =df_trades.loc[df_trades['side']=="buy", 'quantity']
    df_trades['sell_qty']  = df_trades.loc[df_trades['side']=="sell", 'quantity']
    df_trades["sell_value"] = df_trades.loc[df_trades["side"] =="sell","price"].mul(df_trades.loc[df_trades["side"] =="sell","quantity"])
    df_trades["buy_value"] = df_trades.loc[df_trades["side"] =="buy","price"].mul(df_trades.loc[df_trades["side"] =="buy","quantity"])

    gcols = ['week_start_date','client_type','user_id', 'symbol']
    agg_trades_weekly = df_trades.groupby(gcols, as_index=False).agg(
        buy_qty  = ('buy_qty','sum'),
        sell_qty = ('sell_qty','sum'),
        buy_value  = ('buy_value','sum'),
        sell_value = ('sell_value','sum'),
        total_volume = ('total_volume', 'sum'),
        trade_count = ('timestamp', 'count')
    )
    # JUST FOR THIS EXAMPLE Assumption : market price assumed as a price of the most recent transaction on that symbol
    def compute_mark_price(df, symbol_col='symbol', price_col='price', ts_col='timestamp'):
        """
        Compute market price per 'symbol' as the price of the most recent transaction.
        Returns a dict {symbol: price} and writes a 'mark_price' column into df.
        """
        idx = df.groupby(symbol_col)[ts_col].idxmax()
        markt_price = df.loc[idx, [symbol_col, price_col]].set_index(symbol_col)[price_col].astype(float).to_dict()
        df['mark_price'] = df[symbol_col].map(markt_price)
        return markt_price

    # Optional PnL computation
    if compute_pnl:
        markt_price = compute_mark_price(df_trades)
        
        # Weighted average entry price for buys
        agg_trades_weekly['avg_buy_price'] = agg_trades_weekly['buy_value'] / agg_trades_weekly['buy_qty']
        agg_trades_weekly.loc[~np.isfinite(agg_trades_weekly['avg_buy_price']), 'avg_buy_price'] = np.nan  # handle div/0

        # Net position
        agg_trades_weekly['net_qty'] = agg_trades_weekly['buy_qty'] - agg_trades_weekly['sell_qty']
        print(f"net_qty: {agg_trades_weekly['net_qty'].sum(axis=0)}")
        agg_trades_weekly['realized_pnl'] = agg_trades_weekly['sell_value'] - (agg_trades_weekly['avg_buy_price'] * agg_trades_weekly['sell_qty'])

        # Assumption : market price is a price of the most recent transaction on that symbol
        # Map symbol -> price
        agg_trades_weekly['mark_price'] = agg_trades_weekly['symbol'].map(markt_price)

        agg_trades_weekly['unrealized_pnl'] = (agg_trades_weekly['mark_price'] - agg_trades_weekly['avg_buy_price']) * agg_trades_weekly['net_qty']
        agg_trades_weekly['total_pnl'] = agg_trades_weekly['realized_pnl'] + agg_trades_weekly['unrealized_pnl']
    return agg_trades_weekly
    
# ---------- LOAD ----------
def load(df_agg: pd.DataFrame, path: str, table_name: str = "agg_trades_weekly", is_sql = False) -> None:
    """
    Load step: writes the aggregated dataframe to a SQLite database (upsert by replace).
    """
    # Create folder if needed
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    if is_sql:
        db_exists = os.path.exists(path)
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = cur.fetchone() is not None
            # append will create the table if it does not exist
            df_agg.to_sql(table_name, conn, if_exists="append", index=False)
            if db_exists:
                if table_exists:
                    print(f"Appended {len(df_agg)} rows to existing table '{table_name}' in '{path}' (if_exists='append').")
                else:
                    print(f"Database '{path}' existed but table '{table_name}' did not. Created table and wrote {len(df_agg)} rows (if_exists='append').")
            else:
                print(f"Database '{path}' created and table '{table_name}' written with {len(df_agg)} rows (if_exists='append').")
        #with sqlite3.connect(path) as conn:
        #    df_agg.to_sql(table_name, conn, if_exists="replace", index=False)
    else: 
        df_agg.to_excel(path, index=False)

def reporting(
    df_agg: pd.DataFrame,
    client_type: str = "bronze",
    top_n: int = 3,
    metrics: list | None = None,
    compute_pnl: bool = False,
    path: str = "output/top_clients.xlsx",
    add_timestamp: bool = False,
    table_name: str = "agg_trades_weekly" ) -> pd.DataFrame:
    """
    Filter top-N clients of a given client_type by provided metrics and save to Excel via load().
    Returns the dataframe that was saved.
    """

    def plot_weekly_aggregates(df_agg=df_agg, color_map=None, figsize=(10,5)):
        """
        Plot two charts:
        - weekly total_volume (line)
        - weekly stacked bar of trade_count by client_type
        Returns tuple of axes (ax1, ax2).
        """
        if color_map is None:
            color_map = {"bronze":"#CD7F32", "silver":"#C0C0C0", "gold":"#FFD700"}
        #df = df_agg.copy()
        # plot total_volume over time
        ax1 = (
            df_agg.groupby("week_start_date")
            .agg(total_volume=("total_volume", "sum"))
            .reset_index()
            .plot(
                x="week_start_date",
                y="total_volume",
                title="weekly total_volume",
                rot =45,
                xlabel="week_start_date",
                ylabel="total_volume",
                figsize=figsize
            )
        )
        ax1.set_title("Weekly Trade Volume")
        ax1.legend(title="Trade_volume")
        # stacked bar: trade_count by client_type per week
        df_graph2 = df_agg.groupby(["week_start_date", "client_type"])["trade_count"].sum().unstack(fill_value=0)
        colors = [color_map.get(c, "#333333") for c in df_graph2.columns]
        ax2 = df_graph2.plot(
            kind="bar",
            stacked=True,
            color=colors,
            xlabel="week_start_date",
            ylabel="trade_count",
            figsize=figsize
        )
        ax2.set_title("Weekly Trade Count by Client Type")
        ax2.legend(title="client_type")
        
    # Plot
    plot_weekly_aggregates(df_agg)
    plt.show()
    if metrics is None:
        try:
            metrics = ["total_volume", "total_pnl"]
            agg_kwargs = {"total_volume": ("total_volume", "sum")}
            if compute_pnl:
                agg_kwargs["total_pnl"] = ("total_pnl", "sum")
            # aggregate per user
            df_reporting = (
                df_agg[df_agg["client_type"] == client_type]
                .groupby("user_id", as_index=False)
                .agg(**agg_kwargs)
            ).assign(client_type=client_type)
        except: 
            print("Warning: Тільки 'total_volume' та (опційно)  'total_pnl' прийняте до розрахунку")
    # collect top-N for each metric
    top_frames = []
    for m in metrics:
        if m not in df_reporting.columns:
            continue
        top = (
            df_reporting.sort_values(by=m, ascending=False)
            .head(top_n)
            .assign(top_label=m, rank = df_reporting[m].rank(ascending=False))
            .reset_index(drop=True)
        )
        top_frames.append(top)

    df_save = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()

    # optional timestamp in filename
    if add_timestamp:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = f"output/{timestamp}_top_clients.xlsx"

    # persist using existing load() helper
    load(df_save, path, table_name)
    #return df_save

def run_etl(
    input_csv: str = "trades (1) (2) (1).csv",
    sqlite_path: str = "agg_result.db",
    table_name: str = "agg_trades_weekly",
    compute_pnl: bool = True):

    df_extract = extract(input_csv)
    df_transform = transform(df_extract, compute_pnl=compute_pnl)
    load(df_transform, sqlite_path, table_name,is_sql=True)
    reporting(df_transform, compute_pnl = compute_pnl)
    return df_transform

if __name__ == "__main__":
    # Defaults allow ad-hoc running: python etl.py
    out = run_etl(
        input_csv="trades (1) (2) (1).csv",
        sqlite_path="agg_result.db",
        table_name="agg_trades_weekly",
        compute_pnl=True# set True if you provide a market_price.json or a 'pnl' column
    )
    print(out.head(5).to_string(index=True))