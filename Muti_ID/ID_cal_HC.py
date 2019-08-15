# coding:utf-8
import pandas as pd
# import CSI_data
from numpy import *
import tushare as ts
import time

'''
    计算区间的指标：
        1)收益：区间收益率(年化)、Alpha、正收益天数比例;
        2)风险/波动：区间波动率、Beta、负收益天数比例、负收益波动率、最大回撤、峰谷差值、Probability of reaching objective、Efficiency Ratio、Price Density;
        3)收益风险比：Information Ratio、夏普律、Treynor Ratio、Calmar Ratio、Sortino Ratio;
        4)收益曲线形态：Skewness、kurtosis、跟踪误差年化；
'''
class ID_cal:
    # 多支股票数据分析
    def __init__(self):
        # 定义构造器，加载数据
        self.close_weight_data()

    def close_weight_data(self):
        try:
            # 加载收盘数据
            # data01 = open('收盘价.csv')
            data01 = open('close_Back_收盘价.csv')
            self.df = pd.read_csv(data01)
            data02 = open('high_Back_最高价.csv')
            self.high_df = pd.read_csv(data02)
            data03 = open('low_Back_最低价.csv')
            self.low_df = pd.read_csv(data03)
            # 加载权重数据
            file_weight = pd.read_excel('data_CSI&A50.xls')
            self.weights_por = array(file_weight['权重(%)'] / 100)
        except IOError as ioe:
            print(ioe)

    def cal_rate_income(self):
        # 01.计算多支股票的的日收益率-rate_income:R=(P2-P1)/P1,序列
        df = self.df
        data_row, data_col = shape(df)
        income = []
        for i in range(data_row - 1):
            prior_df = mat(df)[data_row - i - 1, :]
            temp_df = mat(df)[data_row - i - 2, :]
            # 收益率-rate_income
            diff_close = temp_df - prior_df
            income.append(diff_close / prior_df)
        d_rate_income = pd.DataFrame(mat(array(income))).fillna(0)
        self.to_csv(filedata=d_rate_income, filename='ID_HC/收益率_close_p.csv')

        # 02.计算年化收益率-复利：[(1+r1)(1+r2)……(1+rN)]**(m/T)-1, r1、r2……rN为日收益率序列
        rows, cols = shape(mat(array(d_rate_income)))
        rate_y = []
        for col in range(cols):
            star_rate = 1.0
            temp_J = mat(d_rate_income)[:, col]
            for row in range(rows):
                val_r = temp_J[row, :]
                temp_val = val_r + 1
                star_rate = temp_val * star_rate
            rate_y.append(star_rate)
        annual_rates = []
        for col in range(cols):
            temp_rate = mat(array(rate_y))[:, col]
            annual_rate = (array(temp_rate) ** (252/data_row))-1
            annual_rates.append(annual_rate)
        annual_rates = mat(array(annual_rates))  # 1,289
        # print('每股的年化收益率：\n', annual_rates, shape(annual_rates))
        an_file = pd.DataFrame(annual_rates)
        an_file.to_csv('ID_HC/年化收益率_复利.csv')

        # 03.计算正收益天数和负收益天数
        # 计算投资组合的收益率(加权)-将每支股票的收益率乘上其对应的权重，得到加权后的股票收益
        # 再对所有股票加权后的收益求和，得到该组合投资的收益
        z_income_day = []
        f_income_day = []
        for j in range(cols):
            rate_income = mat(d_rate_income)[:, j]
            sum_notzeros = 0
            for line in range(rows):
                if rate_income[line, :] > 0:
                    sum_notzeros = sum_notzeros + 1
                    continue
            z_income_day.append(sum_notzeros)
            f_income_day.append(data_row - mat(z_income_day))
        z_income_days_ratio = (mat(z_income_day).T + 1)/data_row
        z_income_days = (mat(z_income_day).T + 1)
        f_income_days_ratio = (f_income_day[-1].T - 1)/data_row
        f_income_days = (f_income_day[-1].T - 1)
        self.to_csv(filedata=z_income_days, filename='ID_HC/每股正收益天数.csv')
        self.to_csv(filedata=z_income_days_ratio, filename='ID_HC/每股正收益天数比例.csv')
        self.to_csv(filedata=f_income_days, filename='ID_HC/每股负收益天数.csv')
        self.to_csv(filedata=f_income_days_ratio, filename='ID_HC/每股负收益天数比例.csv')
        return annual_rates, d_rate_income

    def cal_stdev(self, d_rate_income):
        row, col = shape(d_rate_income)  # 61,289
        # 计算年化波动率: stdev(ri)*sqrt(252), ri：日间收益率
        stdev_ri = d_rate_income.std()  # 每只股票的标准差
        annual_Vt = stdev_ri*sqrt(252)
        self.to_csv(filedata=annual_Vt, filename='ID_HC/年化波动率.csv')
        # 计算年化负收益波动率
        f_annual_VtAll = []
        for j in range(col):
            new_d_rate = []
            temp_rate_j = d_rate_income.ix[:, j]
            for i in range(row):
                temp_rate_i = temp_rate_j.ix[i, :]
                if temp_rate_i <= 0:
                    new_d_rate.append(temp_rate_i)
                    continue
            f_annual_Vt = (mat(new_d_rate).std())*sqrt(252)
            f_annual_VtAll.append(f_annual_Vt)
        f_annual_VtAll = mat(f_annual_VtAll).T
        f_stdev_ri = f_annual_VtAll/sqrt(252)  # 年化负收益的标准差
        self.to_csv(filedata=f_annual_VtAll, filename='ID_HC/负收益波动率(年化).csv')
        return annual_Vt, stdev_ri, f_annual_VtAll, f_stdev_ri

    def cal_mdd(self, d_rate_income):
        # 计算最大回撤
        row, col = shape(d_rate_income)
        top_rowval = array(1.0)
        mdd_all = []
        MDD_all = []
        for j in range(col):
            rate_J = mat(d_rate_income)[:, j]
            values = vstack((top_rowval, array((1+rate_J).cumprod()).T))
            # 回撤值
            D_val = pd.DataFrame(values).cummax()-values
            # d_val = D_val/(D_val+values)
            # 最大回撤
            MDD_val = D_val.max()
            MDD_all.append(MDD_val)
            # 最大回撤率
            # mdd_val = d_val.max()
            # mdd_all.append(mdd_val)
        # print('每股的最大回撤：', MDD_all, shape(MDD_all))
        self.to_csv(filedata=MDD_all, filename='ID_HC/每股的最大回撤.csv')
        return MDD_all

    def cal_beta_alpha(self, d_rate_income):
        # sh_row, ret_sh = self.sh_rate_income()  # 上证指数的收益率
        SH300_d_rate, SH300_weight_por = self.SH300_income()  # HS300的日收益率
        HS_row = shape(SH300_d_rate)[0]
        rate_income = array(d_rate_income)  # 获取每股的收益率
        # 计算HS300的组合收益率
        HS300_por_rate = SH300_d_rate.mul(SH300_weight_por, axis=1).sum(axis=1)

        # 求解上证指数与股票的beta值：beta=cov(A收益率, B收益率)/var(B收益率)；
        # 其中，A为单一金融工具的行情走势，B为一个投资组合或者指数的行情走势；
        # 当0＜β＜1时，单一金融工具行情的波动小于相关的指数波动；
        # 当β＝1时，单一金融工具的行情波动与指数一样；
        # 当β＞1时，单一金融工具行情的波动性要大于指数波动率；
        row, col = shape(rate_income)
        beta = []
        HS300_rate = mat(HS300_por_rate).T
        for j in range(col):
            # ret = hstack((array(ret_sh), array(mat(rate_income[:, j]).T)))
            ret = hstack((array(HS300_rate), array(mat(rate_income[:, j]).T)))
            # beta_single = pd.DataFrame(ret).cov().iat[0, 1]/ret_sh.var()
            beta_single = pd.DataFrame(ret).cov().iat[0, 1] / HS300_rate.var()
            beta.append(beta_single)
        beta = mat(beta).T
        self.to_csv(filedata=beta, filename='ID_HC/beta.csv')

        # 计算Alpha值：投资或基金的绝对回报和按照β系数计算的预期风险回报之间的差额
        # α > 0，表示一基金或股票的价格可能被低估，建议买入。亦即表示该基金或股票以投资技术获得比平均预期回报大的实际回报。
        # α < 0，表示一基金或股票的价格可能被高估，建议卖空。亦即表示该基金或股票以投资技术获得比平均预期回报小的实际回报。
        # α = 0，表示一基金或股票的价格准确反映其内在价值，未被高估也未被低估。亦即表示该基金或股票以投资技术获得平均与预期回报相等的实际回报。
        start_rate_sh = 1
        for k in range(row):
            temp_sh = HS300_rate[k, :]
            temp_val = temp_sh + 1
            start_rate_sh = temp_val * start_rate_sh
        # 获取上证指数的年化收益率-复利:[(1+r1)(1+r2)……(1+rN)]**(m/T)-1
        annual_rate = (array(start_rate_sh) ** (252 / (row+1))) - 1
        alpha = []
        for c in range(col):
            alpha_single = annual_rates[:, c] - annual_rate * beta[c, :]
            alpha.append(alpha_single)
        alpha = mat(array(alpha)).T
        self.to_csv(filedata=alpha, filename='ID_HC/alpha.csv')
        return beta, alpha, HS300_rate

    def cal_sharpe(self, d_rate_income, annual_Vt, annual_rates):
        # 夏普比率 =（年化收益率annual_rates - 无风险收益率） / 年度风险annual_Vt，其中"年度风险"与"年化波动率"等值
        # 如果夏普比率为正值，说明在衡量期内基金的平均净值增长率超过了无风险利率，在以同期银行存款利率作为无风险利率的情况下，说明投资基金比银行存款要好
        # 夏普比率越大，说明基金单位风险所获得的风险回报越高
        # 夏普比率为负时，按大小排序没有意义
        # 计算无风险收益率--短期国债收益率20190725(2:39:48)为3.835
        Rf = (1 + 3.835/100)**(1/360)-1
        Sharpe = (annual_rates.T - Rf)/mat(annual_Vt).T
        self.to_csv(filedata=Sharpe, filename='ID_HC/Sharpe.csv')
        return Rf, Sharpe

    def cal_Information(self, annual_Vt, annual_rates):
        # 计算Information Ratio:
        # 年化收益率(annual_rates)/年度风险(annual_Vt)，其中"年度风险"与"年化波动率"等值
        IR = annual_rates/mat(annual_Vt)
        self.to_csv(filedata=IR, filename='ID_HC/Information.csv')

    def cal_efficiency(self):
        # 计算效益比例：价格变化的净值(正数)/个股价格变化的总和(正数)
        # 或者 ER=|Pt-Pt-n|/sum(Pi-Pi-1),n为测试周期，P为相关金融工具的价格
        # 获取收盘价格
        close_df = self.df
        row, col = shape(close_df)
        # 获取整个周期的初始价格和最后价格的差值，即价格变化的净值(正数)
        diff_Pclose = abs(close_df.ix[row-1, :] - close_df.ix[0, :])
        # 获取个股价格变化的总和(正数)
        sum_close_J = []
        for j in range(col):
            close_df_J = close_df.ix[:, j]
            diff_close_J = []
            for i in range(row-1):
                diff_close_I = abs(close_df_J.ix[row-i-2, :] - close_df_J.ix[row-i-1, :]).round(2)
                diff_close_J.append(diff_close_I)
            sum_diff_close = sum(diff_close_J).round(2)
            sum_close_J.append(sum_diff_close)
        # 每支股票价格变化的净值(正数)
        diff_Pclose = mat(diff_Pclose).T
        # 个股价格变化的总和(正数)
        sum_close_J = mat(sum_close_J).T
        # 效益比例--噪声检测
        ER = pd.DataFrame(diff_Pclose/sum_close_J).fillna(0)
        self.to_csv(filedata=ER, filename='ID_HC/Efficiency_Ratio.csv')

    def cal_Treynor_Calmar(self, Rf, annual_rates, beta, MDD_all):
        # 计算特雷诺比率：(年化收益率:annual_rates-无风险收益率:Rf)/程序的beta系数，疑问：beta系数是否要取绝对值？
        Treynor = (annual_rates.T-Rf)/beta
        self.to_csv(filedata=Treynor, filename='ID_HC/Treynor.csv')
        # 计算卡尔玛比率：年化收益率/最大跌幅，Calmar比率描述的是收益和最大回撤之间的关系。
        # 计算方式为年化收益率(annual_rates)与历史最大回撤(MDD_all)之间的比率。
        # 疑问：最大跌幅是最大回撤(一只股票或基金从价格（净值）最高点到最低点的跌幅)
        # Calmar比率数值越大，基金的业绩表现越好；反之，基金的业绩表现越差。
        Calmar = annual_rates.T/mat(MDD_all)
        self.to_csv(filedata=Calmar, filename='ID_HC/Calmar.csv')

    def cal_Sortino(self, d_rate_income, annual_rates, Rf):
        # 下行风险 = 下行波动率，参照"年度风险"与"年化波动率"等值
        # 计算索迪诺比率：(年化收益率annual_rates-无风险利率Rf)/下行波动率
        # 下行波动率：只衡量低于收益率均值的情况，公式：s_v=[sum(mean_R-ri)**2]/n, mean_R收益率均值，ri各期收益率
        d_rate_income = d_rate_income
        row, col = shape(d_rate_income)
        semivar_all = []
        for j in range(col):
            close_I = []
            close_rate_J = d_rate_income.ix[:, j]
            mean_CRJ = mean(close_rate_J)
            for i in range(row):
                close_rate_I = close_rate_J.ix[i, :]
                if close_rate_I < mean_CRJ:
                    close_I.append(close_rate_I)
                    continue
            close_I = array(close_I).T
            semivar = sum((mean_CRJ - close_I)*(mean_CRJ - close_I))/len(close_I)
            semivar_all.append(semivar)
        # 计算索迪诺比率
        Sortino = (mat(annual_rates).T - Rf)/mat(semivar_all).T
        self.to_csv(filedata=Sortino, filename='ID_HC/Sortino.csv')

    def cal_Skewness_Kurtosis(self):
        # 计算偏度值：[sum(Pi-mean_P)**3]/[(n-1)*std**3],P为价格:收盘价
        # 计算峰度值：[sum(Pi-mean_P)**4]/[(n-1)*std**4]
        close_price = self.df
        row, col = shape(mat(close_price))
        meanP_colse = mat(mean(close_price))
        S_k_all = []; K_all =[]
        for j in range(col):
            diff_all = []
            close_price_J = close_price.ix[:, j]
            mean_PCJ = meanP_colse[:, j]
            for i in range(row):
                close_price_I = close_price_J.ix[row-i-1, :]
                diff = close_price_I - mean_PCJ
                diff_all.append(diff)
            diff_all = (array(diff_all))
            std_Pclose = mat(close_price_J).std()
            # 计算偏度值
            S_k = (sum(diff_all*diff_all*diff_all))/((row-1)*(std_Pclose**3))
            S_k_all.append(S_k)
            # 计算峰度值
            K = (sum(diff_all*diff_all*diff_all*diff_all))/((row-1)*(std_Pclose**4))
            K_all.append(K)
        S_k_all = pd.DataFrame(S_k_all).fillna(0)
        K_all = pd.DataFrame(K_all).fillna(0)
        self.to_csv(filedata=S_k_all, filename='ID_HC/Skewness.csv')
        self.to_csv(filedata=K_all, filename='ID_HC/Kurtosis.csv')

    def cal_Tracking_Error(self, d_rate_income, SH300_d_rate, SH300_weight_por):
        # 计算跟踪误差:以沪深300指数作为基准
        wpor_income = SH300_d_rate.mul(SH300_weight_por, axis=1)
        # 基准组合的收益率
        HS300_por_income = mat(wpor_income.sum(axis=1)).T
        d_rate_income = mat(d_rate_income)
        row, col = shape(d_rate_income)
        # 计算跟踪偏离度:Rpa(i)=Rp(i)-Rb(i)，其中Rp(i)为日收益率，Rb(i)为基准组合收益率
        Rpa = d_rate_income - tile(HS300_por_income, col)
        # self.to_csv(filedata=Rpa, filename='Rpa.csv')
        M_Rpa = Rpa.mean(axis=0)
        # 计算每支股票的年化跟踪误差
        d = Rpa - tile(M_Rpa, (row, 1))
        diff_Rpa = array(Rpa - tile(M_Rpa, (row, 1)))**2
        # self.to_csv(filedata=diff_Rpa, filename='diff_Rpa.csv')
        annual_Tracking_Error = sqrt((252/row)*sum(diff_Rpa, axis=0))
        self.to_csv(filedata=annual_Tracking_Error, filename='ID_HC/annual_Tracking_Error.csv')

    def cal_PriceDensity(self):
        # 计算价格密度：[sum(high_i-low_i)]/[max_high-min_low]
        high_df = self.high_df  # 股票每日最高价数据
        low_df = self.low_df  # 股票每日最低价数据
        diff_HL = mat(high_df-low_df)
        # self.to_csv(filedata=diff_HL, filename='DIFF_HL.csv')
        row, col = shape(diff_HL)
        PD = []
        for c in range(col):
            # 计算每支股票的价格密度
            PD_c = sum(diff_HL[:, c])/(high_df.ix[:, c].max() - low_df.ix[:, c].min())
            PD.append(PD_c)  # 汇总
        PD = pd.DataFrame(PD).fillna(0)
        self.to_csv(filedata=PD, filename='ID_HC/PriceDensity.csv')

    def cal_DIFF_FtoV(self, d_rate_income):
        # 计算峰谷差值
        d_rate = d_rate_income
        row, col = shape(d_rate)
        DIFF_FtoV = []
        for c in range(col):
            d_rate_c = d_rate.ix[:, c]
            max_drate = d_rate_c.max()
            min_drate = d_rate_c.min()
            Diff_FtoV = max_drate-min_drate
            DIFF_FtoV.append(Diff_FtoV)
        DIFF_FtoV = mat(DIFF_FtoV).T
        self.to_csv(filedata=DIFF_FtoV, filename='ID_HC/DIFF_FtoV.csv')
        return DIFF_FtoV

    def SH300_income(self):
        #计算基准组合的的日收益率
        data = open('HS300_close_收盘价.csv')
        df_HS = pd.read_csv(data)
        # 查看哪些列存在缺失值
        df_isnull = list(df_HS.isnull().sum())
        # 缺失值-均值填充
        df_HS = df_HS.fillna(df_HS.mean())
        data_row = shape(df_HS)[0]
        income = []
        for i in range(data_row - 1):
            prior_df = mat(df_HS)[data_row - i - 1, :]
            temp_df = mat(df_HS)[data_row - i - 2, :]
            # 收益率-rate_income
            diff_close = temp_df - prior_df
            income.append(diff_close / prior_df)
        SH300_d_rate = pd.DataFrame(mat(array(income))).fillna(0)
        self.to_csv(filedata=SH300_d_rate, filename='ID_HC/HS300_收益率.csv')
        # 获取SH300的权重数据
        SH300_weight = pd.read_excel('HS300.xlsx')
        weights_por = array(SH300_weight['权重(%)'])
        SH300_weight_por = weights_por/sum(weights_por)
        return SH300_d_rate, SH300_weight_por

    def sh_rate_income(self):
        # 获取上证指数数据
        # df_sh = ts.get_k_data('sh', start="2019-04-16", end="2019-07-16")
        df_sh = ts.get_k_data('sh', start="2017-02-07", end="2019-08-08")  # 对应615日
        sh_close = df_sh['close']
        self.to_csv(filedata=df_sh, filename='ID_HC/上证指数的行情数据.csv')
        sh_row = len(sh_close)
        sh_close = mat(sh_close).T
        rate_income_sh = []
        # 计算上证指数的日收益率
        for i in range(sh_row - 1):
            rate_sh = (sh_close[i + 1, :] - sh_close[i, :]) / sh_close[i, :]
            rate_income_sh.append(rate_sh)
        ret_sh = mat(array(rate_income_sh)).T  # 上证指数的收益率
        return sh_row, ret_sh

    def portfolio_Stock(self, d_rate_income, stdev_ri, f_stdev_ri):
        # 投资组合的ID指标计算--方差var和标准差std除外
        ps_df = pd.read_excel('Portfolio_Stock_HC.xlsx')
        ps_df = mat(ps_df)
        weights_por = mat(self.weights_por).T  # 投资组合权重
        d = tile(weights_por, 21)[:, 0]  # 21个指标
        IDPor_stcck = multiply(ps_df, tile(weights_por, 21)).sum(axis=0)
        self.to_csv(filedata=IDPor_stcck, filename='ID_HC/投资组合的权重加权.csv')

        # 计算投资组合的年化波动率--标准差也被称为波动率
        stdev_ri = mat(stdev_ri).T
        f_stdev_ri = f_stdev_ri  # 负收益波动率
        w_std = multiply(array(weights_por)**2, array(stdev_ri)**2).sum(axis=0)
        fw_std = multiply(array(weights_por)**2, array(f_stdev_ri)**2).sum(axis=0)
        # 收益率之间的相关系数
        corr_ij = mat(pd.DataFrame(d_rate_income).corr())
        # self.to_csv(filedata=stdev_ri, filename='stdev_ri.csv')
        wstd_add = 0
        f_wstd_add = 0
        for std_i in range(shape(weights_por)[0]):
            temp_wi = weights_por[std_i, :]
            for std_j in range(shape(weights_por)[0]):
                if std_i != std_j:
                    temp_wj = weights_por[std_j, :]
                    wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * stdev_ri[std_i, :] * stdev_ri[std_j, :]
                    f_wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * f_stdev_ri[std_i, :] * f_stdev_ri[std_j, :]
                    wstd_add = wstd_multi + wstd_add
                    f_wstd_add = f_wstd_multi + f_wstd_add
                    continue
        por_stdev = (w_std + wstd_add) * sqrt(252)
        f_por_stdev = (fw_std + f_wstd_add) * sqrt(252)
        print('投资组合的年化波动率：', por_stdev, '；投资组合的年化负收益的波动率：', f_por_stdev)
        return corr_ij, wstd_add

    def risk_porStock(self, d_rate_income, HS300_rate):
        top_rowval = array(1.0)
        # 计算加权后：投资组合的收益率-pro_rate
        weights_por = mat(self.weights_por)
        pro_rate = multiply(mat(d_rate_income), weights_por).sum(axis=1)

        # 计算投资组合的最大回撤
        pro_values = vstack((top_rowval, array((1 + mat(pro_rate)).cumprod()).T))
        proD_val = pd.DataFrame(pro_values).cummax() - pro_values
        pro_MDD_val = proD_val.max()
        print('投资组合的最大回撤值：', pro_MDD_val)
        # 计算投资组合的峰谷差值
        pro_DIFF_FtoV = pro_rate.max() - pro_rate.min()
        print('投资组合的的峰谷差值：', pro_DIFF_FtoV)
        # 计算投资组合的Beta值
        ret = hstack((array(HS300_rate), array(mat(pro_rate))))
        pro_beta = pd.DataFrame(ret).cov().iat[0, 1] / HS300_rate.var()
        print('投资组合的Beta值：', pro_beta)
        return pro_MDD_val, pro_DIFF_FtoV, pro_beta

    def cal_wstd_add(self, input_A):
        # 收益率之间的相关系数
        corr_ij = mat(pd.DataFrame(d_rate_income).corr())
        # self.to_csv(filedata=stdev_ri, filename='stdev_ri.csv')
        wstd_add = 0
        weights_por = mat(self.weights_por).T  # 投资组合权重
        for std_i in range(shape(weights_por)[0]):
            temp_wi = weights_por[std_i, :]
            for std_j in range(shape(weights_por)[0]):
                if std_i != std_j:
                    temp_wj = weights_por[std_j, :]
                    # wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * stdev_ri[std_i, :] * stdev_ri[std_j, :]
                    wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * input_A[std_i, :] * input_A[std_j, :]
                    wstd_add = wstd_multi + wstd_add
                    continue
        wstd_add = wstd_add
        return wstd_add

    def to_csv(self, filedata, filename):
        # 文件存储
        tofile = pd.DataFrame(filedata)
        tofile.to_csv(filename)

if __name__ == '__main__':
    start_time = time.clock()
    id_cal = ID_cal()
    annual_rates, d_rate_income = id_cal.cal_rate_income()
    # print('每股的年化收益率：\n', annual_rates, shape(annual_rates))
    annual_Vt, stdev_ri, f_annual_VtAll, f_stdev_ri = id_cal.cal_stdev(d_rate_income)
    MDD_all = id_cal.cal_mdd(d_rate_income)
    beta, alpha, HS300_rate = id_cal.cal_beta_alpha(d_rate_income)
    Rf, Sharpe = id_cal.cal_sharpe(d_rate_income, annual_Vt, annual_rates)
    id_cal.cal_Information(annual_Vt, annual_rates)
    id_cal.cal_efficiency()
    id_cal.cal_Treynor_Calmar(Rf, annual_rates, beta, MDD_all)
    id_cal.cal_Sortino(d_rate_income, annual_rates, Rf)
    id_cal.cal_Skewness_Kurtosis()
    id_cal.cal_PriceDensity()
    DIFF_FtoV = id_cal.cal_DIFF_FtoV(d_rate_income)
    SH300_d_rate, SH300_weight_por = id_cal.SH300_income()
    id_cal.cal_Tracking_Error(d_rate_income, SH300_d_rate, SH300_weight_por)
    corr_ij, wstd_add = id_cal.portfolio_Stock(d_rate_income, stdev_ri, f_stdev_ri)
    id_cal.risk_porStock(d_rate_income, HS300_rate)
    end_time = time.clock()
    print('Run_time:%6fs' % (end_time-start_time))