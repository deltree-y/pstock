import argparse, datetime
from datasets.stock import Stocks
from datasets.stockinfo import StockInfo
from utils.const_def import TOKEN, LATEST_DATE, MIN_TOTAL_MV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sd", default=str(LATEST_DATE), help="Start date for data update")
    parser.add_argument("-ed", default=datetime.datetime.today().strftime('%Y%m%d'), help="End date for data update")
    parser.add_argument("-mmv", default=MIN_TOTAL_MV, help="Minimum market value")
    parser.add_argument("-f", default=0, help="Force download")
    args = parser.parse_args()
    sd = args.sd
    ed = args.ed
    mmv = args.mmv
    f = args.f

    si = StockInfo(TOKEN)
    download_list = si.get_filtered_stock_list(mmv=mmv)
    print(download_list)
    sse = Stocks(download_list, si, start_date=sd, end_date=ed, if_force_download=f)