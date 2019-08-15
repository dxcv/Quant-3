from numpy import *
import tushare as ts

ts.set_token('5e29065d5dbfd27199770371785d6f0c4ae61d026f1af60453dbdf75')
pro = ts.pro_api()
df = pro.daily_basic(ts_code='600000')
print(df)


