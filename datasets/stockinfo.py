#-*- coding:UTF-8 -*-
import tushare as ts
import pandas as pd
import numpy as np
import sys,os,time,logging
from pathlib import Path

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.utils import setup_logging
from utils.tk import TOKEN
from utils.const_def import NAN_FILL, IS_PRINT_TUSHARE_CALL_INFO, INDUSTRY_LIST, STANDARD_DATE, LATEST_DATE
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
        self.data_part11,self.data_part12,self.data_part13,self.data_part14,self.data_part15,self.data_part16 = None,None,None,None,None,None

        self.stock_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "stock_all.csv")
        self.index_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "index_all.csv")
        self.trade_date_list_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "trade_date_all.csv")
        self.stock_list_with_total_mv_fn = os.path.join(BASE_DIR, GLOBAL_DIR, "stock_list_with_total_mv.xlsx")
        self.__update_list()

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

    def update_stock_list_filter(self, mmv=1000000):
        self.__update_list()
        df_mv = self.get_daily_basic(trade_date=str(STANDARD_DATE))
        self.stock_list = pd.merge(self.stock_list, df_mv[['ts_code','total_mv']], how='left', on=['ts_code'])
        self.filtered_list_df = self.stock_list[(self.stock_list['market'] == '主板') \
                                                & (self.stock_list['total_mv'] > float(mmv)) \
                                                    & (self.stock_list['list_date'] <= LATEST_DATE)]
        self.filtered_list_df.to_csv(os.path.join(BASE_DIR, GLOBAL_DIR, "filtered_stocks.csv"), index=False)
        self.filtered_stock_list = self.filtered_list_df['ts_code'].values.tolist()
        logging.info(f"get filtered stock count - < {len(self.filtered_stock_list)} >")

    def get_filtered_stock_list(self, mmv=1000000):
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
                #print(f"INFO: get_start_date() returns {self.index_list.loc[self.index_list['ts_code']==ts_code].list_date.values[0].astype(np.int64)}")
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

    #获取股票所属行业id
    def get_industry_idx(self, ts_code, asset='E'):
        self.__update_list() 
        if asset == 'E':
            try:
                return INDUSTRY_LIST.index(str(self.stock_list.loc[self.stock_list['ts_code']==ts_code].industry.values[0]))
            except:
                logging.error(f"in StockInfo::get_industry().")
                return -1
        else:
            logging.error(f"in StockInfo::get_industry() - 指数不存在所属行业.")
            return -1

    #获取股票id
    def get_stock_idx(self, ts_code, asset='E'):
        self.__update_list() 
        if asset == 'E':
            try:
                return self.stock_list[self.stock_list['ts_code']==ts_code].index[0]
            except:
                raise ValueError(f"未找到目标股票 - <{ts_code}>")
                return -1
        else:
            logging.error(f"in StockInfo::get_stock_idx() - 指数不提供此功能.")
            return -1

    #通用行情接口
    def get_pro_bar(self, ts_code, start_date='', end_date='', asset='E', adj=None):
        start_date = str(start_date) if start_date is not None else None
        end_date = str(end_date) if end_date is not None else None

        if asset == 'I':
            ret = self.ts.pro_bar(ts_code=ts_code, asset='I', start_date=start_date, end_date=end_date)
        elif asset == 'E':
            ret = self.ts.pro_bar(ts_code=ts_code, asset='E', adj=adj, start_date=start_date, end_date=end_date)
        else:
            logging.error(f"in StockInfo::get_pro_bar(). In para error-<asset>")
            exit()
        logging.info(f"Tushare interface - <ts.pro_bar> is running for 1 time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #通用获取tushare数据接口
    def get_tushare_data(self, api_name, remain_columns=None,
                         ts_code=None, trade_date=None, start_date=None, end_date=None):
        start_date = str(start_date) if start_date is not None else None
        end_date = str(end_date) if end_date is not None else None
        
        if api_name in ['shibor', 'us_tycr','us_trycr','us_tbr','us_tltr','us_trltr']:
            date_src_name, date_dst_name = 'date', 'trade_date', 
            one_time_count_limit = 2000
        elif api_name in ['daily', 'moneyflow']:
            one_time_count_limit = 6000
            date_src_name, date_dst_name = 'trade_date', 'trade_date', 
        else:
            one_time_count_limit = -1
            date_src_name, date_dst_name = 'trade_date', 'trade_date', 
        if trade_date is not None:
            ret = getattr(self.pro, api_name)(trade_date=trade_date)
        elif start_date is None and end_date is None:
            ret = getattr(self.pro, api_name)(ts_code=ts_code)
        elif ts_code is not None:
            ret = getattr(self.pro, api_name)(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            ret = getattr(self.pro, api_name)(start_date=start_date, end_date=end_date)
        call_cnt = 1
        if one_time_count_limit > -1:  #有获取数量限制
            next_end_date = self.get_next_end_date(ret, one_time_count_limit, date_src_name)
            while next_end_date is not None:
                if trade_date is not None:
                    new_ret = getattr(self.pro, api_name)(trade_date=trade_date)
                elif start_date is None and end_date is None:
                    new_ret = getattr(self.pro, api_name)(ts_code=ts_code)
                elif ts_code is not None:
                    new_ret = getattr(self.pro, api_name)(ts_code=ts_code, start_date=start_date, end_date=str(next_end_date))
                else:
                    new_ret = getattr(self.pro, api_name)(start_date=start_date, end_date=str(next_end_date))
                new_ret.bfill(inplace=True)
                new_ret.fillna(NAN_FILL,inplace=True)            
                ret = pd.concat([ret, new_ret], ignore_index=True)
                next_end_date = self.get_next_end_date(new_ret, one_time_count_limit, date_src_name)
                call_cnt += 1
        if date_src_name != date_dst_name:
            ret = ret.rename(columns={date_src_name: date_dst_name})
        if remain_columns is not None:
            ret = ret[remain_columns]
        logging.info(f"INFO: Tushare interface - <pro.{api_name}> is running for [{call_cnt}] time.") if IS_PRINT_TUSHARE_CALL_INFO else None
        return ret

    #获取股票对应的总市值（单位：万元）：
    def get_total_mv(self, ts_code):
        daily_basic = self.get_tushare_data(api_name='daily_basic',ts_code=ts_code)
        try:
            ret = daily_basic['total_mv'][0]
        except:
            logging.error(f"in StockInfo::get_total_mv(). ts_code not found - <{ts_code}>")
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

    #若输入日期为交易日,则返回该日期,否则返回前一个交易日
    def get_previous_or_current_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        logging.debug(f"in get_previous_or_current_trade_date(), date={date}, type={type(date)}")
        return date if self.is_trade_date(date) else self.get_pre_trade_date(date)
    
    #若输入日期为交易日,则返回该日期,否则返回后一个交易日
    def get_next_or_current_trade_date(self, date):
        self.__update_list()
        date = type(self.trade_date_list['cal_date'][0])(date)
        logging.debug(f"in get_next_or_current_trade_date(), date={date}, type={type(date)}")
        return date if self.is_trade_date(date) else self.get_next_trade_date(date)

    #获取给定范围的所有交易日数据
    def get_trade_open_dates(self, start_date, end_date):
        start_date, end_date = type(self.trade_date_list['cal_date'][0])(start_date), type(self.trade_date_list['cal_date'][0])(end_date)
        self.__update_list()
        df = self.trade_date_list[(self.trade_date_list['cal_date'] >= start_date) & (self.trade_date_list['cal_date'] <= end_date)]
        if df.empty:
            logging.error(f"No trade date found for <{start_date} - {end_date}>.")
        ret = df[df['is_open']==1]['cal_date']
        ret = pd.DataFrame(ret)
        ret.columns = ['trade_date']
        return ret.astype(str)  #确保trade_date列为字符串类型,20251016修改

    def save_stock_list_with_total_mv(self):
        self.__update_list()
        total_mv_list = []
        for ts_code in self.stock_list['ts_code']:
            total_mv_list.append(self.get_total_mv(ts_code))
            logging.info(f"get total_mv for stock - <{ts_code} : {total_mv_list[-1]}> [{len(total_mv_list)}/{len(self.stock_list['ts_code'])}]")
        tdf = self.stock_list
        tdf['total_mv'] = total_mv_list
        tdf.to_excel(self.stock_list_with_total_mv_fn, sheet_name='stocks')

    def get_top_n_code_group_by_industry(self, n=3):
        # 读取CSV
        df = pd.read_csv(os.path.join(BASE_DIR, GLOBAL_DIR, "stock_list_with_total_mv.csv"), encoding='gbk')
        # 分组并获取每组B列的TOP5
        df['total_mv'] = pd.to_numeric(df['total_mv'], errors='coerce')
        top5 = df[['industry', 'total_mv', 'ts_code']].groupby('industry', group_keys=False).apply(lambda x: x.nlargest(n, 'total_mv'))
        # 获取对应的C列值
        result = top5['ts_code'].tolist()
        print(result)

    def get_next_end_date(self, df, one_time_count_limit, date_col_name='trade_date'):
        if len(df) < one_time_count_limit:
            return None
        ret = self.get_pre_trade_date(df[date_col_name].values[-1])
        return ret

    #获取股票的综合数据
    def get_stock_detail(self, asset='E', ts_code=None, spec_date=None, start_date=None, end_date=None):
        if ts_code is not None: #若ts_code不为空，则默认为取指定stock的数据
            spec_colunm = 'trade_date'
            drop_column = 'ts_code'
            data_basic = self.get_tushare_data(api_name='daily', ts_code=ts_code, start_date=start_date, end_date=end_date)
            data_basic['industry_idx'] = self.get_industry_idx(ts_code) if asset == 'E' else -1
            data_part0 = pd.DataFrame(columns=['trade_date'])
        elif spec_date is not None: #若spec_date不为空，则默认为取指定日期的所有stock数据
            spec_colunm = 'ts_code'
            drop_column = 'trade_date'
            data_basic = self.get_tushare_data(api_name='daily', trade_date=spec_date)
            data_part0 = pd.DataFrame(columns=['ts_code'])
        else:
            logging.error("In para error-<ts_code> and <spec_date> can not be both None.")
            exit()
            
        if asset == 'I':
            return self.get_pro_bar(ts_code=ts_code, asset=asset, start_date=start_date, end_date=end_date)
        elif asset == 'E':
            data_part1 = self.get_tushare_data(api_name='daily_basic', ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date,
                                               remain_columns=['ts_code', 'trade_date', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv']).drop(columns=[drop_column])
            data_part2 = self.get_tushare_data(api_name='moneyflow', ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date,
                                               remain_columns=['trade_date', 'ts_code', 'buy_sm_vol','sell_sm_vol','buy_md_vol','sell_md_vol','buy_lg_vol',\
                                                               'sell_lg_vol','buy_elg_vol','sell_elg_vol','net_mf_vol']).drop(columns=[drop_column])
            self.data_part11 = self.data_part11 if self.data_part11 is not None else self.get_tushare_data(api_name='shibor', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'on', '1w', '6m', '1y'])
            self.data_part12 = self.data_part12 if self.data_part12 is not None else self.get_tushare_data(api_name='us_tycr', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'm1', 'y1', 'y5', 'y10', 'y20', 'y30'])
            self.data_part13 = self.data_part13 if self.data_part13 is not None else self.get_tushare_data(api_name='us_trycr', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'y5', 'y10', 'y30'])
            self.data_part14 = self.data_part14 if self.data_part14 is not None else self.get_tushare_data(api_name='us_tbr', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'w4_bd', 'w4_ce', 'w26_bd', 'w26_ce', 'w52_bd', 'w52_ce'])
            self.data_part15 = self.data_part15 if self.data_part15 is not None else self.get_tushare_data(api_name='us_tltr', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'ltc', 'cmt', 'e_factor'])
            self.data_part16 = self.data_part16 if self.data_part16 is not None else self.get_tushare_data(api_name='us_trltr', start_date=start_date, end_date=end_date, remain_columns=['trade_date', 'ltr_avg'])
            #data_part3 = self.get_margin_detail(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part4 = self.get_block_trade(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part5 = self.get_stk_holdertrade(ts_code=ts_code, trade_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part6 = self.get_share_float(ts_code=ts_code, float_date=spec_date, start_date=start_date, end_date=end_date).drop(columns=[drop_column])
            #data_part7 = self.get_dividend(ts_code=ts_code, ex_date=spec_date).drop(columns=[drop_column])
            data_complete = pd.merge(data_basic, data_part0, how='left', on=[spec_colunm])
            data_complete = pd.merge(data_complete, data_part1, how='left', on=[spec_colunm], suffixes=[None, '_daily_basic'])
            data_complete = pd.merge(data_complete, data_part2, how='left', on=[spec_colunm], suffixes=[None, '_moneyflow'])
            data_complete = pd.merge(data_complete, self.data_part11, how='left', on=[spec_colunm], suffixes=[None, '_shibor'])
            data_complete = pd.merge(data_complete, self.data_part12, how='left', on=[spec_colunm], suffixes=[None, '_us_tycr'])
            data_complete = pd.merge(data_complete, self.data_part13, how='left', on=[spec_colunm], suffixes=[None, '_us_trycr'])
            data_complete = pd.merge(data_complete, self.data_part14, how='left', on=[spec_colunm], suffixes=[None, '_us_tbr'])
            data_complete = pd.merge(data_complete, self.data_part15, how='left', on=[spec_colunm], suffixes=[None, '_us_tltr'])
            data_complete = pd.merge(data_complete, self.data_part16, how='left', on=[spec_colunm], suffixes=[None, '_us_trltr'])
            #data_complete = pd.merge(data_complete, data_part3, how='left', on=[spec_colunm], suffixes=[None, '_margin_detail'])
            #data_complete = pd.merge(data_complete, data_part4, how='left', on=[spec_colunm], suffixes=[None, '_block_trade'])
            #data_complete = pd.merge(data_complete, data_part5, how='left', on=[spec_colunm], suffixes=[None, '_stk_holdertrade'])
            #data_complete = pd.merge(data_complete, data_part6, how='left', on=[spec_colunm], suffixes=[None, '_share_float'])
            #data_complete = pd.merge(data_complete, data_part7, how='left', on=[spec_colunm], suffixes=[None, '_dividend'])
            data_complete.sort_values(by=[spec_colunm], ascending=False, inplace=True)
            data_complete.bfill(inplace=True)
            data_complete.fillna(NAN_FILL,inplace=True)
            return data_complete


if __name__ == "__main__":
    setup_logging()
    ts_code1 = '600036.SH'
    si = StockInfo(TOKEN)
    func_name = 'shibor'
    #df = si.get_tushare_data(api_name=func_name, start_date='19900101', end_date='20240101')
    df = si.get_stock_detail(ts_code=ts_code1, start_date='20220101', end_date='20240101')
    #df = si.get_shibor(start_date='19900101', end_date='20240101')
    #df = si.get_pro_bar(ts_code=ts_code1)
    print(f"rows: {len(df)}")
    print(f"sample data: \n{df.head(3)}")
    #print(f"colunms: {df.columns}")
