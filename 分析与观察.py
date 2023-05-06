import numpy as np
import pandas as pd
import pyecharts
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts.charts import Pie
from pyecharts import options as opts
import collections
import os
#读取数据
data=pd.read_csv('ccf_offline_stage1_test_revised.csv')
#复制数据集
dc=data.copy()
#将Distance中空值填充为-1
dc['Distance'].fillna(-1,inplace=True)
#将领券时间转化成时间类型
dc['Date_received']=pd.to_datetime(dc['Date_received'],format='%Y%m%d')
#判断优惠券是否为满减类型
dc['Ismanjian']=dc['Discount_rate'].map(lambda x:1 if ":" in str(x) else 0)
#将优惠券转换为折扣率
dc['Discount_rate']=dc['Discount_rate'].map(lambda x:round(float(x),5) if ':' not in str(x) else round((float(str(x).split(':')[0])-float(str(x).split(':')[1]))/float(str(x).split(':')[0]),5))
#优惠券折扣小于等于八折的赋值为1，否则为0
dc['Label']=list(map(lambda x: 1 if x<=0.8 else 0,dc['Discount_rate']))
#按日期转化成领券星期
dc["Weekday_receive"]=dc['Date_received'].apply(lambda x: x.isoweekday())
#获取领券月份
dc['Received_month']=dc['Date_received'].apply(lambda x: x.month)
#生成处理后的表格
path='ccf_offline_stage1_test_revised_output'
if not os.path.exists(path):
    os.makedirs(path)
dc.to_csv(path+'ccf_offline_stage1_test_revised_output.csv',index=False)
#用pyecharts对数据做可视化处理
#绘制每天领券数的柱状图
#取出所有领券日期中的非零项
dc_1=dc[dc['Date_received'].notna()]
#按照领券日期进行分类，统计优惠券的数量
group=dc_1.groupby('Date_received',as_index=False)['Coupon_id'].count()
#设置柱状图
Bar_1=(
    Bar(
           init_opts=opts.InitOpts
           (width='1500px',height='600px')
    )
    #设置x轴
    .add_xaxis(list(group['Date_received']))
    #设置y轴
    .add_yaxis('',list(group['Coupon_id']))
    #全局配置
    .set_global_opts(
    #柱状图的标题
            title_opts=opts.TitleOpts(title='每天领券数'),
            #显示图例
            legend_opts=opts.LegendOpts(is_show=True),
#设置横坐标数值，并逆时针旋转60°
xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60,horizontal_align='right'),interval=1),
                         )
                         #其余配置
                        .set_series_opts(
                                opts.LabelOpts(is_show=True),
                                markline_opts=opts.MarkLineOpts(
                                        data=[
                                            opts.MarkLineItem
                                            (type_='max',name='最大值')
                                        ]
                                )
                         )
    )
Bar_1.render(path+'Bar_1.html')
#用Bar绘制各类距离消费次数的柱状图
distance=dc[dc['Distance']!=-1]['Distance'].values
#之前将空数据赋值为-1，此处筛选所有非空数据
distance=dict(collections.Counter(distance))
#先计数非空数据，后存放到新生成的字典里
x=list(distance.keys())
x.sort(reverse=False)#对距离进行排序
y=list(distance.values())
Bar_2=(
    Bar()
    #添加x轴数据
    .add_xaxis(x)
    #添加y轴数据
    .add_yaxis('',y)
    #全局配置
    .set_global_opts(
            title_opts=opts.TitleOpts(title='各类距离消费次数')
    )
    .set_series_opts(
            opts.LabelOpts(is_show=True)
    )
)
Bar_2.render(path+'Bar_2.html')
#绘制消费距离与核销率的柱状图
rate=[dc[dc['Distance']==i]['Label'].value_counts()[1]/dc[dc['Distance']==i]['Label'].value_counts().sum() for i in range (11)]
Bar_3=(
    Bar()
    .add_xaxis(list(range(11)))
    .add_yaxis('核销率',list(rate))
    .set_global_opts(title_opts=opts.TitleOpts(title='消费距离与核销率'))
    .set_series_opts(
            opts.LabelOpts(is_show=False)
    )
)
Bar_3.render(path+'Bar_3.html')
#各种折扣率的领取和核销数量
received=dc[['Discount_rate']]
received['cnt']=1
received=received.groupby('Discount_rate').agg('sum').reset_index()
consume_coupon=dc[dc['Label']==1][['Discount_rate']]
consume_coupon['cnt_2']=1
consume_coupon=consume_coupon.groupby('Discount_rate').agg('sum').reset_index()
data=received.merge(consume_coupon,on='Discount_rate',how='left').fillna(0)
Bar_4=(
    Bar()
    .add_xaxis([float('%.4f'%x) for x in list(data.Discount_rate)])
    .add_yaxis('领取',list(data.cnt))
    .add_yaxis('核销',list(data.cnt_2))
    .set_global_opts(title_opts={'text':'领取与核销'})
    .set_series_opts(
        opts.LabelOpts(is_show=True)
    )
)
Bar_4.render(path+'Bar_4.html')
#折线图
#每周领券数与核销数
#每周核销的优惠券数量
week_coupon=dc[dc['Label']==1]['Weekday_receive'].value_counts()
#每周领取的优惠券数量
week_received=dc[dc['Weekday_receive'].notna()]['Weekday_receive'].value_counts()
week_coupon.sort_index(inplace=True)
week_received.sort_index(inplace=True)
line_1=(
    Line()
    .add_xaxis([str(x) for x in range(1,8)])
    .add_yaxis('领取',list(week_received))
    .add_yaxis('核销',list(week_coupon))
    .set_global_opts(title_opts={'text':'每周领券数与核销数'})
)
line_1.render(path+'line_1.html')
#饼图
#各类优惠券数量占比
v1=['折扣','满减']
v2=list(dc[dc['Date_received'].notna()]['Ismanjian'].value_counts(True))
print(v2)
pie_1=(
    Pie()
    .add('',[list(v) for v in zip(v1,v2)])
    .set_global_opts(title_opts={'text':'各类优惠券数量占比'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}:{c}'))
)
pie_1.render(path+'pie_1.html')
#核销优惠券数量占比
v3=list(dc[dc['Label']==1].Ismanjian.value_counts(True))
pie_2=(
    Pie()
    .add('',[list(v) for v in zip(v1,v3)])
    .set_global_opts(title_opts={'text':'核销优惠券数量占比'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}:{c}'))
)
pie_2.render(path+'pie_2.html')
#正负例
v4=['正例','负例']
v5=list(dc['Label'].value_counts(True))
pie_3=(
    Pie()
    .add('',[list(v) for v in zip(v4,v5)])
    .set_global_opts(title_opts={'text':'正负例'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}:{c}'))
)
pie_3.render(path+'pie_3.html')