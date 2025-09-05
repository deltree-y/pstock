#-*- coding:UTF-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import sys,os,time,logging
from pandas import DataFrame
from pathlib import Path
from datetime import datetime

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.utils import setup_logging
from utils.const_def import TOKEN, NAN_FILL, MIN_TOTAL_MV, LATEST_DATE, STANDARD_DATE, IS_PRINT_TUSHARE_CALL_INFO
from utils.const_def import BASE_DIR, GLOBAL_DIR

#StockInfo用于存储所有股票、指数的基本信息数据，其中
#   stock_list - 存储所有股票基本数据，pandas DataFrame类型
#   index_list - 存储所有指数基本数据，pandas DataFrame类型
#   trade_date_list - 存储所有交易日历数据，pandas DataFrame类型
class StockInfo():
    def __init__(self, token):
        self.ts = ts
        self.ts.set_token(token)
        self.pro = self.ts.pro_api()
        self.stock_list, self.index_list, self.trade_date_list = None, None, None
        self.filtered_list_df = None

        self.stock_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "stock_all.csv")
        self.index_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "index_all.csv")
        self.trade_date_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "trade_date_all.csv")

    def __update_list(self):
        if self.stock_list is not None and self.index_list is not None and self.trade_date_list is not None:
            return
        if os.path.exists(self.stock_list_fn) and os.path.exists(self.index_list_fn) and os.path.exists(self.trade_date_list_fn):
            self.read_stock_list()
        else:
            self.stock_list = self.pro.stock_basic()
            logging.info("Tushare interface - <pro.stock_basic> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
            self.index_list = self.pro.index_basic()
            logging.info("Tushare interface - <pro.index_basic> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
            self.trade_date_list = self.pro.trade_cal(exchange='SSE')
            logging.info("Tushare interface - <pro.trade_cal> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
            self.stock_list.to_csv(self.stock_list_fn, index=False)
            self.index_list.to_csv(self.index_list_fn, index=False)
            self.trade_date_list.to_csv(self.trade_date_list_fn, index=False)

    def save_stock_list(self):
        self.__update_list()
        self.stock_list.to_csv(self.stock_list_fn, index=False)
        self.index_list.to_csv(self.index_list_fn, index=False)
        self.trade_date_list.to_csv(self.trade_date_list_fn, index=False)

    def read_stock_list(self):
        try:
            self.stock_list = pd.read_csv(self.stock_list_fn)
            self.index_list = pd.read_csv(self.index_list_fn)
            self.trade_date_list = pd.read_csv(self.trade_date_list_fn)
        except Exception as e:
            logging.error("读取文件错误.")
            logging.error(e)

    def update_stock_list_filter(self, mmv=MIN_TOTAL_MV):
        self.__update_list()
        df_mv = self.get_daily_basic(trade_date=str(STANDARD_DATE))
        self.stock_list = pd.merge(self.stock_list, df_mv[['ts_code','total_mv']], how='left', on=['ts_code'])
        self.filtered_list_df = self.stock_list[(self.stock_list['market'] == '主板') \
                                                & (self.stock_list['total_mv'] > float(mmv)) \
                                                    & (self.stock_list['list_date'] <= LATEST_DATE)]
        self.filtered_list_df.to_csv(os.path.join(BASE_DIR, GLOBAL_DIR, "filtered_stocks.csv"), index=False)
        self.filtered_stock_list = self.filtered_list_df['ts_code'].values.tolist()
        logging.info(f"get filtered stock count - < {len(self.filtered_stock_list)} >")

    def get_filtered_stock_list(self, mmv=MIN_TOTAL_MV):
        self.update_stock_list_filter(mmv)
        return self.filtered_stock_list

    def print_filtered_stock_list(self):
        self.update_stock_list_filter()
        logging.info(f"filtered stock list is: {self.filtered_stock_list}")

    def get_asset(self, ts_code):
        self.__update_list()
        if len(self.stock_list[self.stock_list['ts_code']==ts_code].values) > 0:
            return 'E'
        elif len(self.index_list[self.index_list['ts_code']==ts_code].values) > 0:
            return 'I'
        else:
            logging.error(f"in StockInfo::get_asset(). ts_code not found - <{ts_code}>")
            exit()

    #获取开盘日期
    def get_start_date(self, ts_code, asset='E'):
        self.__update_list() 
        if asset == 'E':
            try:
                return (self.stock_list.loc[self.stock_list['ts_code']==ts_code].list_date.values[0].astype(np.int64))
            except:
                logging.error(f"in StockInfo::get_start_date().")
                return -1
        elif asset == 'I':
            try:
                return (self.index_list.loc[self.index_list['ts_code']==ts_code].list_date.values[0].astype(np.int64))
            except:
                logging.error(f"in StockInfo::get_start_date().")
                return -1
        else:
            logging.error(f"in StockInfo::get_start_date().")
            return -1


    #获取股票代码对应的股票名称
    def get_name(self, ts_code, asset='E'):
        self.__update_list() 
        if asset == 'E':
            try:
                return str(self.stock_list.loc[self.stock_list['ts_code']==ts_code].name.values[0])
            except:
                logging.error(f"in StockInfo::get_name().")
                return -1
        elif asset == 'I':
            try:
                return str(self.index_list.loc[self.index_list['ts_code']==ts_code].name.values[0])
            except:
                logging.error(f"in StockInfo::get_name().")
                return -1
        else:
            logging.error(f"in StockInfo::get_name().")
    
    #通用行情接口
    def get_pro_bar(self, ts_code, start_date='', end_date='', asset='E', adj=None):
        start_date, end_date = str(start_date), str(end_date)
        if asset == 'I':
            ret = self.ts.pro_bar(ts_code=ts_code, asset='I', start_date=start_date, end_date=end_date)
        elif asset == 'E':
            ret = self.ts.pro_bar(ts_code=ts_code, asset='E', adj=adj, start_date=start_date, end_date=end_date)
        else:
            logging.error(f"in StockInfo::get_pro_bar(). In para error-<asset>")
            exit()
        logging.info(f"Tushare interface - <ts.pro_bar> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #A股日线行情
    #取['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']
    def get_daily(self, ts_code=None, start_date=None, end_date=None, trade_date=None):
        start_date, end_date = str(start_date), str(end_date)
        #cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol']
        if trade_date is not None:
            #ret = self.pro.daily(trade_date=trade_date)[cols]
            ret = self.pro.daily(trade_date=trade_date)
        elif start_date is None and end_date is None:
            #ret = self.pro.daily(ts_code=ts_code)[cols]
            ret = self.pro.daily(ts_code=ts_code)
        else:
            #ret = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)[cols]
            ret = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        logging.info(f"Tushare interface - <pro.daily> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret


    #每日指标， 如换手率、市盈率、市净率、总股本、总市值等
    def get_daily_basic(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        #cols = ['ts_code', 'trade_date', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb', 'ps', 'dv_ratio', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']
        cols = ['ts_code', 'trade_date', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv']
        if trade_date is not None:
            ret = self.pro.daily_basic(trade_date=trade_date)[cols]
        elif start_date is None and end_date is None:
            ret = self.pro.daily_basic(ts_code=ts_code)[cols]
        else:
            ret = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)[cols]
        logging.info(f"Tushare interface - <pro.daily_basic> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #股票历史列表（历史每天股票列表）
    def get_bak_basic(self, ts_code=None, trade_date=None):
        if trade_date is not None:
            ret = self.pro.bak_basic(trade_date=trade_date)
        elif ts_code is not None:
            ret = self.pro.bak_basic(ts_code=ts_code)
        else:
            logging.error("in StockInfo::get_bak_basic().")
            exit()
        logging.info("Tushare interface - <pro.bak_basic> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #备用行情
    def get_bak_daily(self, ts_code=None, trade_date=None):
        if trade_date is not None:
            ret = self.pro.bak_daily(trade_date=trade_date)
        elif ts_code is not None:
            ret = self.pro.bak_daily(ts_code=ts_code)
        else:
            logging.error(f"in StockInfo::get_bak_daily().")
            exit()
        logging.info(f"Tushare interface - <pro.bak_daily> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #获取某一天的所有股票每日指标数据
    def get_one_day_all_daily_basic(self, date):
        ret = self.pro.daily_basic(start_date=date, end_date=date)
        logging.info("INFO: Tushare interface - <pro.daily_basic> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #个股资金流向， 如大/中/小单买入量/金额等
    #数据开始于2010年
    def get_moneyflow(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        cols = ['trade_date', 'ts_code', 'buy_sm_vol','sell_sm_vol','buy_md_vol','sell_md_vol','buy_lg_vol','sell_lg_vol','buy_elg_vol','sell_elg_vol','net_mf_vol']
        if trade_date is not None:
            ret = self.pro.moneyflow(trade_date=trade_date)[cols]
        elif start_date is None and end_date is None:
            ret = self.pro.moneyflow(ts_code=ts_code)[cols]
        else:
            ret = self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)[cols]
        logging.info("INFO: Tushare interface - <pro.moneyflow> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret
    
    #融资融券交易明细
    def get_margin_detail(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        cols = ['trade_date', 'ts_code', 'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye']
        if trade_date is not None:
            ret = self.pro.margin_detail(trade_date=trade_date)[cols]
        elif start_date is None and end_date is None:
            ret = self.pro.margin_detail(ts_code=ts_code)[cols]
        else:
            ret = self.pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date)[cols]
        logging.info("INFO: Tushare interface - <pro.margin_detail> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #大宗交易数据
    def get_block_trade(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        if trade_date is not None:
            block_trade = self.pro.block_trade(trade_date=trade_date)
        elif start_date is None and end_date is None:
            block_trade = self.pro.block_trade(ts_code=ts_code)
        else:
            block_trade = self.pro.block_trade(ts_code=ts_code, start_date=start_date, end_date=end_date)
        logging.info("INFO: Tushare interface - <pro.block_trade> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None

        df = pd.DataFrame(columns=['trade_date','ts_code','block_trade_price', 'block_trade_vol'])
        for row in block_trade.iterrows():
            data = [row[1]['trade_date'],row[1]['ts_code'], 0, 0]
            data[2] = row[1]['price']
            if len(df[df['trade_date']==row[1]['trade_date']]) == 0:#还未存储该日期的数据
                data[3] = row[1]['vol']*10000
                df.loc[len(df)] = data
            else:#已有该日期的数据
                data[3] = df.loc[df['trade_date']==row[1]['trade_date']].values[0][3] + row[1]['vol']*10000
                df.loc[df['trade_date']==row[1]['trade_date']] = data
        return df
    
    #获取股东增减持数据
    def get_stk_holdertrade(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        if trade_date is not None:
            ret = self.pro.stk_holdertrade(trade_date=trade_date)
        elif start_date is None and end_date is None:
            ret = self.pro.stk_holdertrade(ts_code=ts_code)
        else:
            ret = self.pro.stk_holdertrade(ts_code=ts_code, start_date=start_date, end_date=end_date)
        logging.info("INFO: Tushare interface - <pro.stk_holdertrade> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None

        df = pd.DataFrame(columns=['trade_date','ts_code', 'change_vol_G', 'change_vol_P', 'change_vol_C'])
        for row in ret.iterrows():
            data = [row[1]['ann_date'],row[1]['ts_code'],0,0,0]
            holder_type_int = HolderType(row[1]['holder_type']).holder_type_int
            in_de_int = InDe(row[1]['in_de']).in_de_int
            if len(df[df['trade_date']==row[1]['ann_date']]) == 0:#还未存储该日期的数据
                data[holder_type_int+1]=row[1]['change_vol']*in_de_int
                df.loc[len(df)] = data
            else:#已有该日期的数据
                data[2:]=df.loc[df['trade_date']==row[1]['ann_date']].values[0][2:] #检查此处的下标是否正确
                data[holder_type_int+1] += row[1]['change_vol']*in_de_int
                df.loc[df['trade_date']==row[1]['ann_date']] = data
        for row in df.iterrows():
            fixed_date = row[1]['trade_date'] if self.is_trade_date(row[1]['trade_date']) else self.get_pre_trade_date(row[1]['trade_date'])
            df.replace({'trade_date': {row[1]['trade_date']: fixed_date}}, inplace=True)
        
        #简单的replace的话,会造成将替换后的日期与原有的日期重复,形成两条同样日期的数据. 所以此处需要额外再做处理
        pre_date, pre_ts_code = None, None
        for row in df.iterrows():
            if pre_date is not None:
                if row[1]['trade_date'] == pre_date and row[1]['ts_code'] == pre_ts_code:
                    logging.info(f"[stk_holdertrade] 发现<{row[1]['ts_code']}>的重复日期<{row[1]['trade_date']}>数据, 进行容错处理")
                    i = df[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].first_valid_index()
                    data = [row[1]['trade_date'],row[1]['ts_code'],0,0,0]
                    data[2:] = df.loc[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].loc[i][2:]\
                             + df.loc[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].loc[i+1][2:]
                    df[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])] = data
                    df.drop(index=i, inplace=True)
            pre_date, pre_ts_code = row[1]['trade_date'], row[1]['ts_code']
        return df

    #获取限售股解禁数据
    def get_share_float(self, ts_code=None, float_date=None, start_date=None, end_date=None):
        start_date, end_date = str(start_date), str(end_date)
        if float_date is not None:
            share_float = self.pro.share_float(float_date=float_date)
        elif start_date is None and end_date is None:
            share_float = self.pro.share_float(ts_code=ts_code)
        else:
            share_float = self.pro.share_float(ts_code=ts_code, start_date=start_date, end_date=end_date)
        logging.info("INFO: Tushare interface - <pro.share_float> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None

        df = pd.DataFrame(columns=['trade_date','ts_code', 'float_share_open'])
        for row in share_float.iterrows():
            data = [row[1]['float_date'],row[1]['ts_code'], 0]
            if len(df[df['trade_date']==row[1]['float_date']]) == 0:#还未存储该日期的数据
                data[2] = row[1]['float_share']
                df.loc[len(df)] = data
            else:#已有该日期的数据
                data[2] = df.loc[df['trade_date']==row[1]['float_date']].values[0][2] + row[1]['float_share']
                df.loc[df['trade_date']==row[1]['float_date']] = data
        for row in df.iterrows():
            fixed_date = row[1]['trade_date'] if self.is_trade_date(row[1]['trade_date']) else self.get_pre_trade_date(row[1]['trade_date'])
            df.replace({'trade_date': {row[1]['trade_date']: fixed_date}}, inplace=True)
        #简单的replace的话,会造成将替换后的日期与原有的日期重复,形成两条同样日期的数据. 所以此处需要额外再做处理
        pre_date, pre_ts_code = None, None
        for row in df.iterrows():
            if pre_date is not None:
                if row[1]['trade_date'] == pre_date and row[1]['ts_code'] == pre_ts_code:
                    print("INFO: [share_float]发现<%s>的重复日期<%s>数据, 进行容错处理"%(row[1]['ts_code'],row[1]['trade_date']))
                    i = df[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].first_valid_index()
                    data = [row[1]['trade_date'],row[1]['ts_code'],0]
                    data[2:] = df.loc[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].loc[i][2:]\
                             + df.loc[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])].loc[i+1][2:]
                    df[(df['trade_date']==row[1]['trade_date'])&(df['ts_code']==row[1]['ts_code'])] = data
                    df.drop(index=i, inplace=True)
            pre_date, pre_ts_code = row[1]['trade_date'], row[1]['ts_code']
        return df

    #分红送股数据
    def get_dividend(self, ts_code=None, ex_date=None):
        if ex_date is not None:
            dividend = self.pro.dividend(ex_date=ex_date)
        else:
            dividend = self.pro.dividend(ts_code=ts_code)
        logging.info("INFO: Tushare interface - <pro.dividend> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None

        df = pd.DataFrame(columns=['trade_date','ts_code', 'cash_div_tax','stk_div'])
        for row in dividend.iterrows():
            if row[1]['div_proc'] == '实施':
                if len(df[df['trade_date']==row[1]['ex_date']]) == 0:#还未存储该日期的数据
                    df.loc[len(df)] = [row[1]['ex_date'], row[1]['ts_code'], row[1]['cash_div_tax'], row[1]['stk_div']]
        return df
    
    #获取股票对应的总市值（单位：万元）：
    def get_total_mv(self, ts_code):
        daily_basic = self.get_daily_basic(ts_code=ts_code)
        try:
            ret = daily_basic['total_mv'][0]
        except:
            logging.error("ERROR: in StockInfo::get_total_mv().")
            ret = -1
        return ret

    #获取前一交易日的日期
    def get_pre_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        return self.trade_date_list[self.trade_date_list['cal_date']==date]['pretrade_date'].values[0].astype(np.int64)
    
    #获取后一交易日的日期
    def get_next_trade_date(self, date):
        date = type(self.trade_date_list['cal_date'][0])(date)
        return self.__get_next_trade_date(date) if self.is_trade_date(date) else self.__get_next_trade_date(self.get_pre_trade_date(date))

    def __get_next_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        return self.trade_date_list[self.trade_date_list['pretrade_date']==date]['cal_date'].values[0].astype(np.int64)

    #判断是否为交易日
    def is_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        return self.trade_date_list[self.trade_date_list['cal_date']==date]['is_open'].values[0] == 1

    #获取最近前一个交易日的日期
    #若输入日期为交易日,则返回该日期,否则返回前一个交易日
    def get_recent_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        logging.debug(f"in get_recent_trade_date(), date={date}, type={type(date)}")
        return date if self.is_trade_date(date) else self.get_pre_trade_date(date)
    
    #获取给定范围的所有交易日数据
    def get_trade_open_dates(self, start_date, end_date):
        self.__update_list()
        df = self.trade_date_list[(self.trade_date_list['cal_date'] >= start_date) & (self.trade_date_list['cal_date'] <= end_date)]
        if df.empty:
            logging.error(f"No trade date found for <{start_date} - {end_date}>.")
        ret = df[df['is_open']==1]['cal_date']
        ret = pd.DataFrame(ret)
        ret.columns = ['trade_date']
        return ret


    def get_stock_detail(self, asset='E', ts_code=None, spec_date=None, start_date=None, end_date=None):
        if ts_code is not None: #若ts_code不为空，则默认为取指定stock的数据
            spec_colunm = 'trade_date'
            drop_column = 'ts_code'
            #data_basic = self.get_trade_open_dates(start_date=start_date, end_date=end_date)
            data_basic = self.get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            data_part0 = pd.DataFrame(columns=['trade_date'])
        elif spec_date is not None: #若spec_date不为空，则默认为取指定日期的所有stock数据
            spec_colunm = 'ts_code'
            drop_column = 'trade_date'
            data_basic = self.get_daily(trade_date=spec_date)
            data_part0 = pd.DataFrame(columns=['ts_code'])
        else:
            logging.error("In para error-<ts_code> and <spec_date> can not be both None.")
            exit()
            
        if asset == 'I':
            return self.get_pro_bar(ts_code=ts_code, asset=asset, start_date=start_date, end_date=end_date)
        elif asset == 'E':
            data_part1 = self.get_daily_basic(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            data_part2 = self.get_moneyflow(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part3 = self.get_margin_detail(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part4 = self.get_block_trade(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part5 = self.get_stk_holdertrade(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part6 = self.get_share_float(ts_code=ts_code, float_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part7 = self.get_dividend(ts_code=ts_code, ex_date=spec_date).drop(columns=[drop_column])
            data_complete = pd.merge(data_basic, data_part0, how='left', on=[spec_colunm])
            data_complete = pd.merge(data_complete, data_part1, how='left', on=[spec_colunm], suffixes=[None, '_daily_basic'])
            data_complete = pd.merge(data_complete, data_part2, how='left', on=[spec_colunm], suffixes=[None, '_moneyflow'])
            #data_complete = pd.merge(data_complete, data_part3, how='left', on=[spec_colunm], suffixes=[None, '_margin_detail'])
            #data_complete = pd.merge(data_complete, data_part4, how='left', on=[spec_colunm], suffixes=[None, '_block_trade'])
            #data_complete = pd.merge(data_complete, data_part5, how='left', on=[spec_colunm], suffixes=[None, '_stk_holdertrade'])
            #data_complete = pd.merge(data_complete, data_part6, how='left', on=[spec_colunm], suffixes=[None, '_share_float'])
            #data_complete = pd.merge(data_complete, data_part7, how='left', on=[spec_colunm], suffixes=[None, '_dividend'])
            data_complete.sort_values(by=[spec_colunm], ascending=False, inplace=True)
            data_complete.bfill(inplace=True)
            data_complete.fillna(NAN_FILL,inplace=True)
            return data_complete


class HolderType():
    def __init__(self, holder_type):
        self.holder_type = holder_type
        self.holder_type_int = -1
        self.__get_inter_class()

    def __get_inter_class(self):
        if self.holder_type == 'G':
            self.holder_type_int = 1
        elif self.holder_type == 'P':
            self.holder_type_int = 2
        elif self.holder_type == 'C':
            self.holder_type_int = 3
        else:
            logging.error(f"unexcept <holder_type> -[{self.holder_type}]")
            exit()

class InDe():
    def __init__(self, in_de):
        self.in_de = in_de
        self.in_de_int = self.__get_inter_class()
    
    def __get_inter_class(self):
        if self.in_de not in ['IN', 'DE']:
            logging.error(f"InDe() unexcept <in_de> -[{self.in_de}]")
            exit()
        ret = 1 if self.in_de == 'IN' else -1
        return ret

if __name__ == "__main__":
    setup_logging()
    ts_code1 = '600036.SH'
    ts_code2 = '000001.SH'
    si = StockInfo(TOKEN)
    si.test()

    #print("code:<%s>, name:<%s>, open date:<%s>. "%(ts_code1, si.get_name(ts_code1), si.get_start_date(ts_code1)))
    #print("code:<%s>, name:<%s>, open date:<%s>. "%(ts_code2, si.get_name(ts_code2,asset='I'), si.get_start_date(ts_code2,asset='I')))
    #daily_basic = si.get_total_mv(ts_code1)
    if 0:
        d = {'ts_code':[], 'total_mv':[]}
        df = pd.DataFrame(data=d)

        total_mv_list = []
        for ts_code in si.stock_list['ts_code']:
            total_mv_list.append(si.get_total_mv(ts_code))
            print(ts_code)
        tdf = si.stock_list
        tdf.insert(10,'total_mv',total_mv_list)

        #print(tdf)
        tdf.to_excel("data\\stocks.xlsx", sheet_name='stocks')
    

    #ret = si.get_daily_basic(trade_date='20250821')
    #ret = si.get_trade_open_dates('20250701','20250822')
    #ret = ret['trade_date'].values
    
    #print(ret)
    #print(type(ret))
    #si.print_filtered_stock_list()
