import numpy as np
import pandas as pd

df=pd.read_csv('ccf_offline_stage1_test_revised.csv',sep=',')
#统计测试集中总记录数量
record_num=df.shape[0]
#统计用户数量
User_num=len(df['User_id'].value_counts())
#统计商家数量
merchat_num=len(df['Merchant_id'].value_counts())
#统计领取优惠券记录的数量
receive_num=df['Date_received'].count()
#统计消费距离类别数量
distance=len(df['Distance'].value_counts())
#统计优惠券种类
variety_num=len(df['Coupon_id'].value_counts())
#将其转化成时间类型
df['Date_received']=pd.to_datetime(df['Date_received'],format='%Y%m%d')
#统计最早/最晚领券时间
first_receive_time=df['Date_received'].min()
latest_receive_time=df['Date_received'].max()
#将统计数据按照规范格式输出
print("记录数量: ",record_num)
print("用户数量: ",User_num)
print("商家数量: ",merchat_num)
print("领券数量: ",receive_num)
print("消费距离类别: ",distance)
print("优惠券种类: ",variety_num)
print("最早领券时间: ",first_receive_time)
print("最晚领券时间: ",latest_receive_time)
