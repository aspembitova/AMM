import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from homotopy import *



inputs = ConfigParser()
inputs["Params"] = {
    "x_price": 1, 
    "y_price": 3000,
    "peg_price": 3000,
    "x_centre": 1000000000,
    "y_centre": 333333,
    "x_coord": 1000000000,
    "y_coord": 333333,
    "fee": 0.005,
    "c": 0.3,
    "y_excess": 0,
    "x_excess": 0,
    "threshold": 0.03,
    "tolerance": 0.02,
    "low": 1,
    "high": 1000000000, 
    "marginal_price": 3000
}

d = {'amount0': [3, 5, 10, 15, 1, -5, -8, -3, 5], 'price': [3000, 2995, 3000, 3002, 3002, 2998, 2995, 2993, 3000]}
test_set1 = pd.DataFrame(data=d)

with open('config.ini', 'w') as config_object:
    inputs.write(config_object)

config_object = ConfigParser()
config_object.read("config.ini") 

keys = ["x_price", "y_price", "peg_price", "x_centre", "y_centre", "x_coord", "y_coord", "fee", "c", "y_excess", "x_excess", "threshold", 
       "tolerance", "low", "high", "marginal_price"]

amounts = test_set1['amount0']
market_prices = test_set1['price']
x_price, y_price, peg_price, x_centre, y_centre, x_coord, y_coord, fee_p, c, y_excess, x_excess, threshold, tolerance, low, high, marginal_price = [float(config_object.get("Params",x)) for x in keys]
sample = np.arange(1, x_centre)

m_prices = [] 
cs = []
pys = []
y_prices = []
dxs = []
arb_amounts = []
mps = [] 

for amount, market_price in zip(amounts, market_prices): 
    
    
    if (abs(y_price-peg_price))/peg_price >= threshold:  #re-peg condition met, we change to new price
        
        if y_price < peg_price: #price went down
        
            if marginal_price < y_price: # marginal price is lower than peg and oracle, change to oracle:
                #x_centre, c_y = new_centre(x_centre, c, peg_price, y_price)
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee_p)
                peg_price = y_price
            
            elif marginal_price > peg_price: # marginal price is higher than peg and oracle, stay at peg:
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, peg_price, x_excess, y_excess, fee_p)
                peg_price = peg_price
                    
            else: # marginal price in between peg and oracle, change to marginal
                #x_centre, c_x0 = new_centre(x_centre, c, peg_price, marginal_price)
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, marginal_price, x_excess, y_excess, fee_p)
                peg_price = marginal_price
            

        else: #price went up (y_price>peg_price)
            if marginal_price > y_price: #marg price higher than new oracle price and peg price, change to new oracle price
                x_centre, c_x = new_centre(x_centre, c, peg_price, y_price) #change the x centre and c
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c_x, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee_p)
                peg_price = y_price # we change peg price to the oracle price
                c = c_x
                
            elif marginal_price < peg_price: 
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, peg_price, x_excess, y_excess, fee_p)
                peg_price = peg_price 

            else:
                x_centre, c_x0 = new_centre(x_centre, c, peg_price, marginal_price)
                marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, marginal_price, x_excess, y_excess, fee_p)
                peg_price = marginal_price
        fees = 0
            
    else: #re-peg condition not met, use old price
        marginal_price, x_coord, y_coord, x_excess, y_excess, fee1 = swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, peg_price, x_excess, y_excess, fee_p)
        fees += fee1 
 
    #Arbitrage transactions to match the market price:  
    if market_price == marginal_price:
        
        dx = 0 #no arbitrage needed
        
    elif market_price>marginal_price: 
        dx = arbitrage_binary_search(x_coord, y_coord, x_price, y_price, market_price, c, low, high, tolerance)
            
    else: 
        target_price_x = 1/market_price
        dx = -arbitrage_binary_search(y_coord, x_coord, y_price, x_price, target_price_x, c, low, high, tolerance)
    
    arb_amount = dx - abs(amount)
    
    if arb_amount<0:
        marginal_price, x_coord, y_coord, x_excess, y_excess, fee0 = swapIn(arb_amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee_p)
    else: 
        marginal_price, x_coord, y_coord, x_excess, y_excess, fee0 = swapOut(arb_amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee_p)   
    high = x_centre
    dxs.append(dx)
    arb_amounts.append(arb_amount)
    m_prices.append(marginal_price)
    pys.append(peg_price)
    cs.append(c) 
    times = [1] * len(m_prices)
    mps.append(market_price)
    y_price = twap(m_prices, times, 10)
    y_prices.append(y_price)
