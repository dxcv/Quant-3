# coding:utf-8
from numpy import *
import pandas as pd
import tushare as ts
from sklearn.svm import SVR
from xgboost import XGBRegressor

class hc_data:
    # 获取20150101-20190808的股票数据和回测后的数据
    # 回测close、high、low数据
    def load_data(self):
        '''
           加载全部股票数据
        '''
        read_file = pd.read_excel('data_CSI&A50.xls')
        ts_code_list = list((read_file['代码'].dropna()))
        # 回测的数据
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
                self.to_csv(res_data, 'close_res_data')
                # self.to_csv(res_data, 'high_res_data')
                # self.to_csv(res_data, 'low_res_data')
                count += 1
                print(count, 'run：', single_code)
            except IOError as e:
                print('导入数据有bug!')

    def fill_mean(self):
        # 数据清理: 缺失值-均值填充
        res_data_file = pd.read_csv('res_data.csv')
        res_data_new = res_data_file.fillna(res_data_file.mean())
        self.to_csv(res_data_new, 'res_data_new')

        all_data = mat(res_data_new)  # 13*289
        return all_data

    def build_model(self, all_data):
        # 数据分离-切片处理，数据回测：close
        stock_data_all = all_data
        single_col = 13
        m, n = shape(stock_data_all)
        init_yclose = array(zeros(m))
        close_x = []
        for i in range(289):
            stock_i = stock_data_all[:, (single_col * i + 1):(single_col * (i + 1) + 1)]
            # 1.close
            # stock_close_x = stock_i[:, 0:-1]
            # stock_close_y = stock_i[:, -1]
            # 2.high=1,low=2
            for k in range(single_col):
                if k != 2:
                    close_x.append(stock_i[:, k])
            stock_close_x = mat(array(close_x)).T
            stock_close_y = stock_i[:, 2]
            # 数据回测--回测模型XGBoost
            ystock_pridict = self.XGBoost_model(stock_close_x, stock_close_y)
            init_yclose = vstack((init_yclose, ystock_pridict))
            print('回测第', i, '支股票')
        pridict_yclose = mat(init_yclose).T
        close_yp = pridict_yclose[:, 1:290]
        # self.to_csv(close_yp, 'close_Back_Test')
        # 数据倒序排列-按最新时间到旧时间排序
        close_p = []
        for line in reversed(range(m)):
            temp_line = close_yp[line, :]
            close_p.append(temp_line)
        close_p = mat(array(close_p))
        # self.to_csv(close_p, 'close_Back')
        self.to_csv(close_p, 'low_Back')
        # self.to_csv(close_p, 'low_Back')
        return close_p

    def XGBoost_model(self, x, y):
        '''
        构建XGBoost回归模型
        '''
        xgb = XGBRegressor()
        xgb.fit(x, y)
        y_pridict = array(xgb.predict(x)).T
        return y_pridict

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
        # 1.close
        stock_pcolse = stock_df[['open', 'high', 'low', 'volume', 'price_change', 'p_change',
                                 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'close']]
        self.to_csv(stock_pcolse, single_code)
        return stock_df, stock_pcolse, single_code

    def to_csv(self, filedata, code_name):
        # 文件存储
        try:
            tofile = pd.DataFrame(filedata)
            filename = 'HcData/' + code_name + '.csv'
            tofile.to_csv(filename)
        except IOError as ie:
            print(ie)

if __name__ == '__main__':
    hc = hc_data()
    # hc.load_data()
    all_data = hc.fill_mean()
    hc.build_model(all_data)