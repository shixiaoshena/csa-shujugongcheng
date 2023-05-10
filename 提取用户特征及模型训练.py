import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

off_train = pd.read_csv("ccf_offline_stage1_train.csv")
off_test = pd.read_csv("ccf_offline_stage1_test_revised.csv")

# 数据预处理
def preprocess(received):
    data=pd.DataFrame(received)
    print("preprocessing")
    ## 时间格式转换
    data["date_received"]=pd.to_datetime(data["Date_received"],format="%Y%m%d")
    if 'Date' in data.columns.tolist():
        data["date"]=pd.to_datetime(data["Date"],format="%Y%m%d")
    ## 满减格式规范
    ### 满减转换成折扣率 <注意仅当rate中有':'>
    def rate_to_count(rate):
        rate = str(rate)
        a, b = rate.split(sep=":")
        return (int(a)-int(b))/int(a)
    ### have_discount = 是否有优惠券
    data.insert(loc=4, column="have_discount",value=data["Discount_rate"].notnull())
    # print(off_train["have_discount"])
    ### discount = 折扣率
    data.insert(loc=5, column="discount",value=data["Discount_rate"].map(lambda x: float(rate_to_count(x)) if ":" in str(x) else float(x)))
    # print(off_train["discount"])
    ### discount = 是否为满减
    data.insert(loc=5, column="is_manjian", value=data["Discount_rate"].map(lambda x: True if ":" in str(x) else False))
    ### 满减门槛价格
    data.insert(loc=6,column="price_before",value=data["Discount_rate"].map(lambda x: float(str(x).split(sep=":")[0]) if ":" in str(x) else -1))
    # print(off_train["price_before"])
    ## 缺失值处理
    data["Distance"]=data["Distance"].map(lambda x: -1 if x!=x else x)
     ## 统一数据格式
    data["Distance"]=data["Distance"].map(int)

    return data

def next_date_series(data=pd.DataFrame(), column='column_not_found'):
    users=data[['User_id']].groupby('User_id').agg(sum)
    users['next']=np.nan
    sorted_data=data[['User_id', column]]
    sorted_data['temp_index']=list(range(len(sorted_data)))
    sorted_data=sorted_data.sort_values(by=[column,'temp_index'],ascending=False)
    sorted_data['next_'+column]=np.nan
    # print(sorted_data)
    for index,item in sorted_data.iterrows():
        user_id=int(item['User_id'])
        # print(index)
        sorted_data['next_'+column][index]=users['next'][user_id]
        users['next'][user_id]=item[column]
    sorted_data=sorted_data.sort_index()
    next_series=sorted_data['next_'+column].copy()
    # print(next_series)
    return next_series


def get_user_feature_label(received):
    '''
    提取标签区间的用户特征
    - 保证传入的所有数据都有领券
    '''
    print("getting user_feature_label")
    received['cnt']=1
    name_prifix="u_l_" #前缀

    # 1、用户当月领券数
    pivoted=pd.pivot_table(
        received,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    received=pd.merge(received,pivoted,on='User_id',how='left')
    # 用户当月领取相同券数
    pivoted=pd.pivot_table(
        received,index=['User_id', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"this_coupon"})
    received=pd.merge(received,pivoted,on=['User_id', 'Coupon_id'],how='left')
    # 用户当天领券数
    print(received)
    pivoted=pd.pivot_table(
        received,index=['User_id', 'date_received'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"today_coupon"})
    received=pd.merge(received,pivoted,on=['User_id', 'date_received'],how='left')
    # 平均每天领券数

    # 用户是否当天领取同一券

    # 当天领取相同券数
    pivoted=pd.pivot_table(
        received,index=['User_id', 'date_received', 'Coupon_id'], values='cnt',aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"today_this_coupon"})
    received=pd.merge(received,pivoted,on=['User_id', 'date_received', 'Coupon_id'],how='left')
    
    # 领取满减优惠券数
    
    # 领取折扣优惠券数

    # 时间窗口截止日期距离领券日期的天数

    # 是否第一次/最后一次领券
    keys=['User_id']
    pivoted = received[['User_id','Date_received']].sort_values(['Date_received'],ascending = True)
    fir = pivoted.drop_duplicates(keys,keep = 'first')
    fir[name_prifix+'if_is_first_received'] = 1
    received = pd.merge(received,fir,on = ['User_id']+['Date_received'],how = 'left')
    received.fillna(0,downcast = 'infer',inplace = True)

    pivoted = received[['User_id','Date_received']].sort_values(['Date_received'],ascending = False)
    la = pivoted.drop_duplicates(keys,keep = 'first')
    la[name_prifix+'if_is_last_received'] = 1
    received = pd.merge(received,la,on = ['User_id']+['Date_received'],how = 'left')
    received.fillna(0,downcast = 'infer',inplace = True)

    # 用户之前领取的相同券数

    # 用户距上次领券时间间隔
    
    '''
    # 用户距下次领券时间间隔
    pivoted = data[["User_id","Date_received"]]
    pivoted["date_received_next"]=next_date_series(pivoted,"Date_received")
    print(pivoted)
    print(pivoted[pivoted['date_received_next'].notnull()])

    input()

    pivoted[name_prifix+'timedelta_rec_use'] = received_used["Date"]-received_used["Date_received"]
    pivoted=pivoted.groupby("User_id").agg(np.mean).reset_index()
    print(pivoted)
    data=pd.merge(data,pivoted,on='User_id',how='left')
    data[name_prifix+'timedelta_rec_use']=data[name_prifix+'timedelta_rec_use'].map(lambda x: int(x) if x==x else -1)
    print(data[name_prifix+'timedelta_rec_use'])
    '''
    received.drop(['cnt'], axis=1, inplace=True)

    ######################## 排序特征 ########################
    name = 'User_id'
    # 折扣率排序
    received[name_prifix + 'discount_rate_rank'] = \
        received.groupby(name)['discount'].rank(ascending = False)
    received[name_prifix + 'discount_rate_rank_ascend'] = \
        received.groupby(name)['discount'].rank(ascending = True)
    # 距离排序
    received[name_prifix + 'distance_rank'] = \
        received.groupby(name)['Distance'].rank(ascending = False)
    received[name_prifix + 'distance_rank_ascend'] = \
        received.groupby(name)['Distance'].rank(ascending = True)
    # 领券日期排序
    received[name_prifix + 'date_received_rank'] = \
        received.groupby(name)['Date_received'].rank(ascending = False)
    received[name_prifix + 'date_received_rank_ascend'] = \
        received.groupby(name)['Date_received'].rank(ascending = True)
    # # 满减门槛价格排序
    # received[name_prifix + 'price_before_rank'] = \
    # received.groupby(name)['price_before'].rank(ascending = False)
    # received[name_prifix + 'price_before_rank_ascend'] = \
    # received.groupby(name)['price_before'].rank(ascending = True)
    return received

def get_merchant_feature_label(received):
    '''
    提取标签区间的商家特征
    - 保证传入的所有数据都有领券
    '''
    print("getting merchant_feature_label")
    received['cnt']=1
    name_prifix="m_l_" #前缀
    
    # 发放优惠券数
    pivoted=received.groupby('Merchant_id').agg({'cnt': sum}).reset_index() \
        .rename(columns={'cnt': name_prifix+"howmuch_coupon"})
    received=pd.merge(received,pivoted,on='Merchant_id',how='left')

    # 发放优惠券种类数
    pivoted=received[['Merchant_id', 'Coupon_id','cnt']].copy().drop_duplicates()
    pivoted=pivoted.groupby('Merchant_id').agg({'cnt': sum}).reset_index() \
        .rename(columns={'cnt': name_prifix+"howmuch_kindof_coupon"})
    received=pd.merge(received,pivoted,on='Merchant_id',how='left')

    # 当天发券数
    pivoted=received.groupby(['Merchant_id', 'date_received']).agg({'cnt': sum}).reset_index() \
        .rename(columns={'cnt': name_prifix+"howmuch_coupon_today"})
    received=pd.merge(received,pivoted,on=['Merchant_id', 'date_received'],how='left')

    # 当天发券的优惠券种数
    pivoted=received[['Merchant_id', 'Coupon_id', 'date_received','cnt']].copy().drop_duplicates()
    pivoted=pivoted.groupby(['Merchant_id', 'date_received']).agg({'cnt': sum}).reset_index() \
        .rename(columns={'cnt': name_prifix+"howmuch_kindof_coupon_today"})
    received=pd.merge(received,pivoted,on=['Merchant_id', 'date_received'],how='left')

    # (n天内)领券数

    ######################## 排序特征 ########################
    name = 'Merchant_id'
    # 折扣率排序
    received[name_prifix + 'discount_rate_rank'] = \
        received.groupby(name)['discount'].rank(ascending = False)
    received[name_prifix + 'discount_rate_rank_ascend'] = \
        received.groupby(name)['discount'].rank(ascending = True)
    # 距离排序
    received[name_prifix + 'distance_rank'] = \
        received.groupby(name)['Distance'].rank(ascending = False)
    received[name_prifix + 'distance_rank_ascend'] = \
        received.groupby(name)['Distance'].rank(ascending = True)
    # 领券日期排序
    received[name_prifix + 'date_received_rank'] = \
        received.groupby(name)['Date_received'].rank(ascending = False)
    received[name_prifix + 'date_received_rank_ascend'] = \
        received.groupby(name)['Date_received'].rank(ascending = True)
    # # 满减门槛价格排序
    # received[name_prifix + 'price_before_rank'] = \
    # received.groupby(name)['price_before'].rank(ascending = False)
    # received[name_prifix + 'price_before_rank_ascend'] = \
    # received.groupby(name)['price_before'].rank(ascending = True)
    return received

def get_coupon_feature_label(received):
    '''
    提取标签区间的优惠券特征
    - 保证传入的所有数据都有领券
    '''
    print("getting user_feature_label")
    received['cnt']=1
    name_prifix="c_l_" #前缀

    # 当月领券数
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_received"})
    received=pd.merge(received,pivoted,on='Coupon_id',how='left')

    ######################## 排序特征 ########################
    name = 'Coupon_id'
    # 折扣率排序
    received[name_prifix + 'discount_rate_rank'] = \
        received.groupby(name)['discount'].rank(ascending = False)
    received[name_prifix + 'discount_rate_rank_ascend'] = \
        received.groupby(name)['discount'].rank(ascending = True)
    # 距离排序
    received[name_prifix + 'distance_rank'] = \
        received.groupby(name)['Distance'].rank(ascending = False)
    received[name_prifix + 'distance_rank_ascend'] = \
        received.groupby(name)['Distance'].rank(ascending = True)
    # 领券日期排序
    received[name_prifix + 'date_received_rank'] = \
        received.groupby(name)['Date_received'].rank(ascending = False)
    received[name_prifix + 'date_received_rank_ascend'] = \
        received.groupby(name)['Date_received'].rank(ascending = True)
    
    return received


def get_user_feature_history(received, concat_on):
    '''
    提取历史区间(需要标签数据)的用户特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 稍后处理缺失值中-1的影响
    '''
    print("getting user_feature_history")
    received=received.copy()
    received['cnt']=1
    name_prifix = "u_h_" #前缀
    
    # 1、用户领券数
    pivoted=pd.pivot_table(
        received,index='User_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    # 2、用户领券"并消费"数
    pivoted=pd.pivot_table(
        received,index='User_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    # 3、用户领券"未消费"数
    concat_on[name_prifix+'not_use']=concat_on[name_prifix+'total_coupon']-concat_on[name_prifix+"used"]
    # 4、 用户领券"并消费"数 / 领券数、
    concat_on[name_prifix+'rate_used']=concat_on[name_prifix+"used"]/concat_on[name_prifix+'total_coupon']
    # 对前四个特征进行缺失值处理
    concat_on[name_prifix+"total_coupon"] = concat_on[name_prifix+"total_coupon"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"used"] = concat_on[name_prifix+"used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"not_use"] = concat_on[name_prifix+"not_use"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rate_used"] = concat_on[name_prifix+"rate_used"].map(lambda x: -1 if x!=x else x)

    received_used = received[received['label']==1].copy() # 数据中有消费的部分
    used_have_distance=received_used[received_used['Distance']!=-1]

    # 5、用户领取"并消费"优惠券的平均折扣率、
    pivoted=pd.pivot_table(
        received_used,index='User_id', values='discount', aggfunc=np.average
            ).reset_index().rename(columns={'discount': name_prifix+"used_average_discount"})
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    concat_on[name_prifix+"used_average_discount"] = \
        concat_on[name_prifix+"used_average_discount"].map(lambda x: -1 if x!=x else x)

    # 6、用户领取"并消费"优惠券的平均距离、
    pivoted=pd.pivot_table(
        used_have_distance,index='User_id', values='Distance', aggfunc=np.average
            ).reset_index().rename(columns={'Distance': name_prifix+"used_average_distance"})
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    concat_on[name_prifix+"used_average_distance"] = \
        concat_on[name_prifix+"used_average_distance"].map(lambda x: -1 if x!=x else int(x))

    # 7、用户在多少不同商家领取"并消费"优惠券
    gpb=received_used.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"used_howmuch_merchant"})
    concat_on=pd.merge(concat_on,gpb,on='User_id',how='left')
    # 8、用户在多少不同商家领取优惠券
    gpb=received.groupby('User_id').agg({'Merchant_id':lambda x:len(set(x))}) \
        .reset_index().rename(columns={'Merchant_id': name_prifix+"howmuch_merchant"})
    concat_on=pd.merge(concat_on,gpb,on='User_id',how='left')
    # 9、用户在多少不同商家领取"并消费"优惠券 / 用户在多少不同商家领取优惠券
    concat_on.insert(0,name_prifix+'rate_used_howmuch_merchant',
        concat_on[name_prifix+"used_howmuch_merchant"]/concat_on[name_prifix+'howmuch_merchant'])
    # 对特征7-9进行缺失值处理
    concat_on[name_prifix+"used_howmuch_merchant"] = \
        concat_on[name_prifix+"used_howmuch_merchant"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"howmuch_merchant"] = \
        concat_on[name_prifix+"howmuch_merchant"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rate_used_howmuch_merchant"] = \
        concat_on[name_prifix+"rate_used_howmuch_merchant"].map(lambda x: -1 if x!=x else x)
    # 用户用券购买平均距离
    pivoted=used_have_distance[['User_id','Distance']].groupby('User_id').agg(np.mean)
    pivoted=pivoted.reset_index().rename(columns={'Distance': name_prifix+"used_mean_distance"})
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    concat_on[name_prifix+'used_mean_distance']=concat_on[name_prifix+'used_mean_distance'].map(lambda x: int(x) if x==x else -1)
    # 用户领券到消费平均时间间隔
    pivoted = received_used[["User_id"]]
    pivoted[name_prifix+'timedelta_rec_use'] = received_used["Date"]-received_used["Date_received"]
    pivoted=pivoted.groupby("User_id").agg(np.mean).reset_index()
    # print(pivoted)
    concat_on=pd.merge(concat_on,pivoted,on='User_id',how='left')
    concat_on[name_prifix+'timedelta_rec_use']=concat_on[name_prifix+'timedelta_rec_use'].map(lambda x: int(x) if x==x else -1)

    received.drop(['cnt'], axis=1, inplace=True)
    return concat_on

def get_coupon_feature_history(received, concat_on):
    '''
    提取历史区间(需要标签数据)的优惠券特征
    - 保证传入的所有数据都有领券
    - 保证传入的数据应有label列
    - 分数下降: 2/3/4
    '''
    print("getting coupon_feature_history")
    received=received.copy()
    received['cnt']=1
    name_prifix = "c_h_" #前缀
    # print(received)
    # 1、当前种类优惠券被领取数
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    concat_on=pd.merge(concat_on,pivoted,on='Coupon_id',how='left')
    # 2、当前种类优惠券被领取"并消费"数 分数下降
    pivoted=pd.pivot_table(
        received,index='Coupon_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"used"})
    concat_on=pd.merge(concat_on,pivoted,on='Coupon_id',how='left')
    # 3、当前种类优惠券被领取"未消费"数 分数下降
    concat_on.insert(0,name_prifix+'not_use',concat_on[name_prifix+'total_coupon']-concat_on[name_prifix+"used"])
    # 4、 当前种类优惠券被领取"并消费"数 / 领取数 分数下降
    concat_on.insert(0,name_prifix+'rate_used',concat_on[name_prifix+"used"]/concat_on[name_prifix+'total_coupon'])
    # 1-4缺失值处理
    concat_on[name_prifix+'total_coupon'] = concat_on[name_prifix+'total_coupon'].map(lambda x: int(x) if x==x else -1)
    concat_on[name_prifix+'used'] = concat_on[name_prifix+'used'].map(lambda x: int(x) if x==x else -1)
    concat_on[name_prifix+'not_use'] = concat_on[name_prifix+'not_use'].map(lambda x: int(x) if x==x else -1)
    concat_on[name_prifix+'rate_used'] = concat_on[name_prifix+'rate_used'].map(lambda x: x if x==x else -1)

    # 根据优惠券折扣率等提取特征

    received.drop(['cnt'], axis=1, inplace=True)
    return concat_on

def get_merchant_feature_history(arg_received, arg_all, concat_on):
    '''
    提取历史区间的商家特征
    - 保证传入的所有数据都有领券
    - 把history的领券数删除
    '''
    print("getting coupon_feature_history")
    data_received=arg_received.copy()
    # data_all=arg_all.copy()
    data_received['cnt']=1
    name_prifix = "m_h_" #前缀

    # 发放优惠券数
    pivoted=pd.pivot_table(
        data_received,index='Merchant_id', values='cnt', aggfunc=np.sum
            ).reset_index().rename(columns={'cnt': name_prifix+"total_coupon"})
    concat_on=pd.merge(concat_on,pivoted,on='Merchant_id',how='left')
    # 15天内被领券购买量
    pivoted=pd.pivot_table(
        data_received,index='Merchant_id', values='label', aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"rec_used"})
    concat_on=pd.merge(concat_on,pivoted,on='Merchant_id',how='left')
    # 15天内核销率
    concat_on[name_prifix+'rate_used']=concat_on[name_prifix+"rec_used"]/concat_on[name_prifix+'total_coupon']

    # 缺失值处理
    concat_on[name_prifix+"total_coupon"] = \
        concat_on[name_prifix+"total_coupon"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rec_used"] = \
        concat_on[name_prifix+"rec_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"rate_used"] = \
        concat_on[name_prifix+"rate_used"].map(lambda x: -1 if x!=x else x)

    data_received.drop(['cnt'], axis=1, inplace=True)

    return concat_on

def get_user_merchant_feature_history(arg_received, arg_all, concat_on):
    '''
    提取历史区间(需要标签数据)商家和用户特征
    - arg_received:有领券的集合, arg_all: 全体集合
    - 保证传入的数据应有label列
    - features将合并到data_received
    - 稍后处理缺失值中-1的影响
    '''
    print("getting user_merchant_feature_history")
    data_rec=arg_received.copy()[['User_id','Merchant_id','label','Distance']]
    data_all=arg_all.copy()[['User_id','Merchant_id','label','Distance']]
    data_rec['cnt']=1
    data_all['cnt']=1
    name_prifix = "um_h_" #前缀

    # 用户在店家领券并消费数 分数下降
    pivoted=pd.pivot_table(
        data_rec,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_rec_used"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    # 用户在店家消费数
    pivoted=pd.pivot_table(
        data_all,index=['User_id', 'Merchant_id'], values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_merchant_used"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    # 率
    user_rec_used = pd.pivot_table(
        data_rec,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_rec_used"})
    concat_on=pd.merge(concat_on, user_rec_used, on='User_id', how='left')
    
    user_used = pd.pivot_table(
        data_all,index='User_id', values='label',aggfunc=np.sum
            ).reset_index().rename(columns={'label': name_prifix+"user_used"})
    concat_on=pd.merge(concat_on, user_used, on='User_id', how='left')
    
    # 用户在此商家领券购买数/领券购买总数 分数下降
    concat_on[name_prifix+"user_merchant_rec_used_rate"] = \
        concat_on[name_prifix+"user_merchant_rec_used"] / concat_on[name_prifix+"user_rec_used"]
    # 用户在此商家购买数/购买总数 分数下降
    concat_on[name_prifix+"user_merchant_used_rate"] = \
            concat_on[name_prifix+"user_merchant_used"] / concat_on[name_prifix+"user_used"]

    concat_on=concat_on.drop([name_prifix+"user_rec_used", name_prifix+"user_used"],axis=1)

    concat_on[name_prifix+"user_merchant_rec_used"] = \
        concat_on[name_prifix+"user_merchant_rec_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"user_merchant_used"] = \
        concat_on[name_prifix+"user_merchant_used"].map(lambda x: 0 if x!=x else int(x))
    concat_on[name_prifix+"user_merchant_rec_used_rate"] = \
        concat_on[name_prifix+"user_merchant_rec_used_rate"].map(lambda x: -1 if x!=x else x)
    concat_on[name_prifix+"user_merchant_used_rate"] = \
        concat_on[name_prifix+"user_merchant_used_rate"].map(lambda x: -1 if x!=x else x)

    # # 用户与店家平均距离 分数下降
    used_have_distance=data_rec[data_rec['Distance']!=-1]
    pivoted=pd.pivot_table(
        used_have_distance,index=['User_id', 'Merchant_id'], values='Distance',aggfunc=np.mean
            ).reset_index().rename(columns={'Distance': name_prifix+"user_merchant_mean_distance"})
    concat_on=pd.merge(concat_on,pivoted,on=['User_id','Merchant_id'],how='left')

    concat_on[name_prifix+"user_merchant_mean_distance"] = \
        concat_on[name_prifix+"user_merchant_mean_distance"].map(lambda x: -1 if x!=x else int(x))

    return concat_on


def get_label(data):
    # 数据打标
    total_seconds = 15*24*3600.0
    print(data)
    data['label']=list(map(lambda x,y: 
        1 if (x - y).total_seconds()<=total_seconds else 0, data['date'], data['date_received']))
    return data
 
def get_dataset(field):
    '''
    构造数据集
    '''
    dataset=pd.DataFrame(field).copy()

    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)

    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))

    return dataset

def model_xgb(train, validate, to_test=True, big_train=True):
    '''
    训练模型
    - return (model: xgb_booster)
    '''
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1.1,
              'gamma': 0.1,
              'lambda': 10,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
            #   'tree_method': 'gpu_hist',
              'scale_pos_weight': 26}

    if to_test==False:
        dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
        dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
        watchlist = [(dtrain, 'train'),(dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
    elif big_train==True:
        big_train = pd.concat([train, validate], axis=0)
        big_train.to_csv('result/features/big_train.csv')
        dtrain = xgb.DMatrix(big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=big_train['label'])
        watchlist = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, num_boost_round=1900, evals=watchlist)
    else:
        dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
        dval = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=validate['label'])
        watchlist = [(dval, 'train'),(dtrain, 'val')]
        model = xgb.train(params, dval, num_boost_round=500, evals=watchlist)

    return model

def interval(data,contain,beginning,lenth):
    '''
    划分区间
    '''
    field = data[
        data[contain].isin(pd.date_range(beginning, periods=lenth))].copy()
    return field

def get_feat_importance(model):
    # 特征重要性
    feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    feature_importance['feature'] = model.get_score().keys()
    feature_importance['importance'] = model.get_score().values()
    feature_importance.sort_values(['importance'], ascending=False, inplace=True)
    feature_importance.to_csv('result/feat_importance.csv', index=False, header=None)
    print("feature_importance is saved.")

def get_result(model,test):
    '''
    返回线上测试集的预测结果
    - args (model: xgb_booster, test: DataFrame)
    - returns (result, feat_importance)
    '''
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['pred'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    result.to_csv('result/result.csv', index=False, header=None)
    print("results are saved.")

def get_feature_for(history_field, all_his_field, label_field, filename=None):
    # 提取特征
    label_field=get_user_feature_label(label_field)
    label_field=get_merchant_feature_label(label_field)
    label_field=get_coupon_feature_label(label_field)
    label_field=get_user_feature_history(history_field, label_field)
    label_field=get_coupon_feature_history(history_field, label_field)
    label_field=get_merchant_feature_history(history_field,all_his_field,label_field) # 不对劲
    label_field=get_user_merchant_feature_history(history_field,all_his_field,label_field) # 不对劲

    if filename != None:
        print(filename)
        label_field.to_csv('result/features/'+filename+'.csv')
    # 构造数据集
    label_field=get_dataset(label_field)
    return label_field



off_train = preprocess(off_train)
off_train = get_label(off_train)

# print(len(off_train[off_train['label']==1]))
# print(len(off_train[off_train['label']==0]))
# input()

off_test = preprocess(off_test)

# 数据划分
train_history_field=interval(off_train,'date_received','2016/1/31',90).copy()
all_history_field_t=interval(off_train,'date','2016/1/31',90).copy()
train = interval(off_train,'date_received','2016/4/30',31).copy() # 训练集

validate_history_field=interval(off_train,'date_received','2016/3/2',90).copy()
all_history_field_v=interval(off_train,'date','2016/3/2',90).copy()
validate = interval(off_train,'date_received','2016/5/31',31).copy() # 验证集

test_history_field=interval(off_train,'date_received','2016/4/1',90).copy()
all_history_field_test=interval(off_train,'date','2016/4/1',90).copy()
test = off_test # 测试集

print(off_train)
print(train)
print('train')
train=get_feature_for(train_history_field, all_history_field_t, train, 'train')
validate=get_feature_for(validate_history_field, all_history_field_v, validate)
print('valid')
print(test)
test=get_feature_for(test_history_field, all_history_field_test, test)
print('test')

# 训练
to_test= True
model = model_xgb(train, validate, to_test=to_test, big_train=True)
get_feat_importance(model)
# model.save_model("model/")

if to_test==True:
    get_result(model,test)

### end ###