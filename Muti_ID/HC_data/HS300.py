# coding:utf-8
from numpy import *
import pandas as pd
import tushare as ts

class hc_data:
    # 获取20150101-20190808的股票数据
    def load_data(self):
        '''
           加载全部股票数据
        '''
        read_file = pd.read_excel('HS300.xlsx')
        ts_code_list = list((read_file['股票代码'].dropna()))
        self.file_handle(ts_code_list)

    def get_ts_data(self, single_code):
        # 获取数据：20150101-20190808
        stock_df = ts.get_hist_data(single_code, start='2015-01-01', end='2019-08-08')
        return stock_df

    def file_handle(self, ts_code_list):
        res_data = pd.DataFrame()
        count = 0
        for i in range(len(ts_code_list)):
            try:
                single_code = str(round(ts_code_list[i]))
                stock_df, stock_pcolse, single_code = self.profile_name(count, single_code)
                # 合并所有股票数据--数据合并
                all_stock = stock_pcolse
                res_data = pd.concat([res_data, all_stock], axis=1)
                self.to_csv(res_data, 'HS300_res_data')
                count += 1
                print(count, 'run：', single_code)
            except IOError as e:
                print('HS300导入数据有bug!')

    def fill_mean(self):
        # 数据清理: 缺失值-均值填充
        res_data_file = pd.read_csv('HS300_res_data.csv')
        res_data_new = res_data_file.fillna(res_data_file.mean())
        self.to_csv(res_data_new, 'HS300_res_data_new')
        all_data = mat(res_data_new)  # 13*289
        return all_data

    def get_data(self, all_data):
        # 数据分离-切片处理
        stock_data_all = all_data
        single_col = 13
        m, n = shape(stock_data_all)
        # init_yclose = array(zeros((m, 1)))
        stock_close = []
        for i in range(300):
            stock_i = stock_data_all[:, (single_col*i + 1):(single_col*(i+1)+1)]
            # init_yclose = vstack((init_yclose, array(stock_i)))
            stock_i_close = stock_i[:, -1]
            print('回测第', i, '支HS300股票')
            stock_close.append(stock_i_close)
        stock_close = mat(array(stock_close)).T
        # 数据倒序排列-按最新时间到旧时间排序
        close_p = []
        for line in reversed(range(m)):
            temp_line = stock_close[line, :]
            close_p.append(temp_line)
        close_p = mat(array(close_p))
        self.to_csv(close_p, 'HS300_close_Back')
        return close_p

    def profile_name(self, count, single_code):
        if len(single_code) == 1:
            single_code = '00000' + str(single_code)
            stock_df = self.get_ts_data(single_code)
        if len(single_code) == 2:
            single_code = '0000' + str(single_code)
            stock_df = self.get_ts_data(single_code)
        if len(single_code) == 3:
            single_code = '000' + str(single_code)
            stock_df = self.get_ts_data(single_code)
        if len(single_code) == 4:
            single_code = '00' + str(single_code)
            stock_df = self.get_ts_data(single_code)
        if len(single_code) == 5:
            single_code = '0' + str(single_code)
            stock_df = self.get_ts_data(single_code)
        if len(single_code) == 6:
            single_code = str(single_code)
            stock_df = self.get_ts_data(single_code)
        else:
            print('数据加载错误！')
        # 数据整理：close作为因变量，其它的特征作为自变量
        stock_pcolse = stock_df[['open', 'high', 'low', 'volume', 'price_change', 'p_change',
                                 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'close']]
        self.to_csv(stock_pcolse, single_code)
        return stock_df, stock_pcolse, single_code

    def to_csv(self, filedata, code_name):
        # 文件存储
        try:
            tofile = pd.DataFrame(filedata)
            filename = 'HS300_Data/' + code_name + '.csv'
            tofile.to_csv(filename)
        except IOError as ie:
            print(ie)

if __name__ == '__main__':
    hc = hc_data()
    # hc.load_data()
    all_data = hc.fill_mean()
    hc.get_data(all_data)