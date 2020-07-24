import numpy as np
import pandas as pd

from utils import portfolio
import yfinance as yf 


class CryptoEnvironment:
    
    def __init__(self, prices = './data/crypto_portfolio.csv', capital = 1e6):       
        self.prices = prices  
        self.capital = capital  
        self.data = self.load_data()

    def load_data(self):
        data =  pd.read_csv(self.prices)
        try:
            data.index = data['Date']
            data = data.drop(columns = ['Date'])
        except:
            data.index = data['date']
            data = data.drop(columns = ['date'])            
        return data
    
    def preprocess_state(self, state):
        return state
    
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        
        assert lookback <= t
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t, alpha = 0.01):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])

        data_period = self.data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
      
        sharpe = local_portfolio(returns, weights)[-1]
        sharpe = np.array([sharpe] * len(self.data.columns))          
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]
        
        return np.dot(returns, weights), rew
        


class ETFEnvironment:
     

    def __init__(self, prices = pd.read_csv('stock_price.csv', index_col=0),returns = pd.read_csv('stock_return.csv', index_col=0), capital = 1e6):
      
        self.returns = returns
        self.prices = prices   
        self.capital = capital
        self.data = self.load_data()

    def load_data(self):
        returns= self.returns
        prices= self.prices
        return prices
        
        #以上这些部分负责加载数据

        
        

    def preprocess_state(self, state):
        return state
        #这里的t和lookback直接从头开始数数就完事了，跟时间半毛钱关系没有   
    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False):
        # 怎里可以挑选用哪一个作为state，直接用时间序列，也可以用价格变化百分比协方差矩阵
        assert lookback <= t
        #单纯的判断一下，不符合就报错
        
        decision_making_state = self.data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()
        # 出一个协方差矩阵，lookback这个时间段，各个股票之间的协方差
        if is_cov_matrix:
            x = decision_making_state.cov()
            return x

        #他把这段时间每种股票的协方差矩阵作为state了
        else:
            if is_raw_time_series:
                decision_making_state = self.data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)
        # 这里的意思是有两种输入模式

        
 
  
    
    
    
    
    
   


    def get_reward(self, action, action_t, reward_t):
        
        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() * 252
            covs = returns.cov() * 252
            P_ret = np.sum(rets * weights)# portfolio总共的return，一个数
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))# portfolio 总共的方差，一个数
            P_sharpe = (P_ret-0.0125)/ (P_vol) # sharp ratio，我给减了一个无风险利率
            return np.array([P_ret, P_vol, P_sharpe])
        # .dot 如果处理的是一维数组，则得到的是两数组的內积
        # 如果是二维数组（矩阵）之间的运算，则得到的是矩阵积
        
        weights = action #分配portoflio就是这个强化学习的action，动作，做动作的时间，得到奖励的时间 产生奖励 sharp ratio 再做动作
        returns = self.data[action_t:reward_t].pct_change().dropna()
        # 只算了action之后直到在一个action的award
        
        rew = local_portfolio(returns, weights)[-1]
        # 这里奖励的是更大的sharp ratio，如果我往括号里放-1，如果我放了-3，只追求最大的return
        rew = np.array([rew] * len(self.data.columns))
        
        return np.dot(returns, weights), rew
        # 给出的是 portfolio return 和 reward （sharp ratio）
        