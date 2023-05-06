import pandas as pd
import numpy as np
import os
#读取数据
data=pd.read_csv('ccf_offline_stage1_test_revised.csv')
#复制数据集
dc=data.copy()
#查看数据集中的缺失值
print(dc.isnull().any)
#缺失值比例查看
print("数据集的数据缺失率为：\n",dc.isnull().sum()/len(dc))
#将Distance中的空值填充为-1
dc['Distance'].fillna(-1,inplace=True)
#格式化领券时间
dc['Date_received']=pd.to_datetime(dc['Date_received'],format='%Y%m%d')
#时间格式转换
dc['Weekday_receive']=dc['Date_received'].apply(lambda x:x.isoweekday())
#获取领券月份
dc['Receive_month']=dc['Date_received'].apply(lambda x:x.month)
#判断优惠券类型
dc['Ismanjian']=dc['Discount_rate'].map(lambda x: 1 if ":" in str(x) else 0)
#满减优惠券的折扣率
dc['Discount_rate']=dc['Discount_rate'].map(lambda x:round(float(x),5) if ':' not in str(x) else round((float(str(x).split(':')[0])-float(str(x).split(':')[1]))/float(str(x).split(':')[0]),5))
#数据打标：优惠券折扣大于等于八折的为1，反之为0
dc['Label']=list(map(lambda x: 1 if x<=0.8 else 0,dc['Discount_rate']))
#生成处理后的数据表格
path='ccf_offline_stage1_test_revised_output'
if not os.path.exists(path):
    os.makedirs(path)
dc.to_csv(path+'ccf_offline_stage1_test_revised_output.csv',index=False)