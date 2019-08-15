# coding:utf-8

import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class ID_plot:
    # 指标趋势
    def __init__(self):
        # 加载指标数据
        self.id_df = pd.read_excel('ID_CSI&A50.xls')
        file = open('[figure]收益率.csv')
        self.income_rate = pd.read_csv(file)

    def plot_IDdata(self):
        # 获取数据，并形成趋势图
        an_inrate = self.income_rate.ix[:, 1]
        an_inrate.plot(kind='line', label='民生银行', legend=True, marker='.')
        plt.xlim(1, 61)
        plt.title('Stock: 民生银行_日收益率')
        plt.show()


if __name__ == '__main__':
    id_plot = ID_plot()
    id_plot.plot_IDdata()