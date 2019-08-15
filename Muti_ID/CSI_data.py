# coding:utf-8
import pandas as pd
import tushare as ts

class CSI:
    # CSI&A50 中华通股票组合统计分析
    def __init__(self):
        path_api = '5e29065d5dbfd27199770371785d6f0c4ae61d026f1af60453dbdf75'
        ts.set_token(path_api)
        self.pro = ts.pro_api()
        # self.load_data()

    def load_data(self):
        '''
           加载全部股票数据
        '''
        # read_file = pd.read_excel('data_CSI&A50.xls')
        read_file = pd.read_excel('HS300.xlsx')
        # # ts_code_list = list((read_file['代码'].dropna()))
        ts_code_list = list((read_file['股票代码'].dropna()))
        stock_colse, stock_high, stock_low = self.file_handle(ts_code_list)
        # 实际数据
        # self.to_csv(filedata=stock_colse, code_name='HS300_收盘价')
        # self.to_csv(filedata=stock_high, code_name='HS300_最高价')
        # self.to_csv(filedata=stock_low, code_name='HS300_最低价')
        return stock_colse

    def get_ts_data(self, single_code):
        # 获取数据
        self.df = ts.get_hist_data(single_code, start='2015-01-01', end='2019-08-08')
        self.to_csv(self.df, single_code)
        return self.df

    def file_handle(self, ts_code_list):
        count = 0
        stock_colse = pd.DataFrame()
        stock_high = pd.DataFrame()
        stock_low = pd.DataFrame()
        for i in range(len(ts_code_list)):
            single_code = str(round(ts_code_list[i]))
            strs = single_code
            if len(single_code) == 1:
                single_code = '00000' + str(single_code)
                self.get_ts_data(single_code)
            if len(single_code) == 2:
                single_code = '0000' + str(single_code)
                self.get_ts_data(single_code)
            if len(single_code) == 3:
                single_code = '000' + str(single_code)
                self.get_ts_data(single_code)
            if len(single_code) == 4:
                single_code = '00' + str(single_code)
                self.get_ts_data(single_code)
            if len(single_code) == 5:
                single_code = '0' + str(single_code)
                self.get_ts_data(single_code)
            if len(single_code) == 6:
                single_code = str(single_code)
                self.get_ts_data(single_code)
            else:
                print('数据加载错误！')
                break
            count += 1
            print(count, 'run：', single_code)
            # 获取每支股票的收盘价-真实数据
            stock_colse[i] = self.df['close']
            # 获取每支股票的最高价和最低价
            stock_high[i] = self.df['high']
            stock_low[i] = self.df['low']
        return stock_colse, stock_high, stock_low


    def to_csv(self, filedata, code_name):
        # 文件存储
        try:
            tofile = pd.DataFrame(filedata)
            # filename = 'data01/' + code_name + '.csv'
            filename = 'HS_data_HC/' + code_name + '.csv'
            tofile.to_csv(filename)
        except IOError as ie:
            print(ie)


if __name__ == '__main__':
    csi = CSI()
    csi.load_data()