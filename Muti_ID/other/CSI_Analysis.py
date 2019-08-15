# coding:utf-8
import pandas as pd
# import CSI_data
from numpy import *

'''
    计算区间的指标：
        1)收益：区间收益率(年化)、Alpha、正收益天数比例;
        2)风险/波动：区间波动率、Beta、负收益天数比例、负收益波动率、最大回撤、峰谷差值、Probability of reaching objective、Efficiency Ratio、Price Density;
        3)收益风险比：Information Ratio、夏普率、Treynor Ratio、Calmar Ratio、Sortino Ratio;
        4)收益曲线形态：Skewness、kurtosis、跟踪误差年化；
'''
class CSI_AlS:
    # 多支股票数据分析
    def __init__(self):
        self.csi_als_data()

    def csi_als_data(self):
        # 加载数据
        data = open('收盘价.csv')
        self.df = pd.read_csv(data)
        # print(self.df)

    def cal_income(self):
        # 计算收益率--用最终资金减去初始资金就是所获得的盈利，然后用盈利除以初始资金就是收益率
        df = self.df
        row = shape(df)[0]
        income = []
        for i in range(row-1):
            prior_df = mat(df)[i, :]
            temp_df = mat(df)[i+1, :]
            # 收益率-rate_income
            diff_close = temp_df - prior_df
            income.append(diff_close/prior_df)
        rate_income = array(income)
        # rate_return = self.df.pct_change().dropna()
        # rate_return = rate_return.drop([0])
        # self.to_csv(filedata=rate_return, filename='z收益率.csv')
        # 计算年化收益率:Y=(收益率+1)**(D/T)-1
        rate_income_year = (rate_income + 1)**(250/62) - 1
        mean_yrate_income = (pd.DataFrame(mean(rate_income_year, axis=0))).fillna(0)
        self.to_csv(filedata=(mean_yrate_income), filename='y收益率_均值.csv')
        # print('年化收益率:', mean_yrate_income)
        # all_mean = mean(mat(mean_yrate_income), axis=1)

        file_weight = pd.read_excel('data_CSI&A50.xls')
        weights_stock = file_weight['权重(%)']
        weights_stock = mat(weights_stock).T  # (289,1)
        all_ymean_rate = mat(mean_yrate_income).T
        w_rate_income_year = pd.DataFrame(all_ymean_rate).mul(array(weights_stock))
        return_rate = mean(w_rate_income_year)


        dd =0



        # print('年化收益率-mean:', all_mean)
        return rate_income

    def Test_single_stock(self):
        dfs = self.df
        sig_df = mat(dfs['600000']).T
        row = shape(sig_df)[0]
        incomes = []
        for i in range(row-1):
            prior_dfs = mat(sig_df)[i, :]
            temp_dfs = mat(sig_df)[i+1, :]
            # 收益率-rate_income
            diff_closes = temp_dfs - prior_dfs
            incomes.append(diff_closes/prior_dfs)
        rate_incomes = array(incomes)
        # 计算年化收益率:Y=(收益率+1)**(D/T)-1
        rate_income_years = (rate_incomes + 1) ** (250 / 62) - 1
        mean_y = mean(rate_income_years)
        # print('单支股票年收益率：', mean_y)






    def to_csv(self, filedata, filename):
        # 文件存储
        tofile = pd.DataFrame(filedata)
        tofile.to_csv(filename)


if __name__ == '__main__':
    csi = CSI_AlS()
    # csi.csi_als_data()
    rate_income = csi.cal_income()
    # print(rate_income)
    csi.Test_single_stock()