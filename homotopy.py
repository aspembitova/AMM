import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

#Functions to calculate amounts out knowing amount in: 
def getConstantSum(dx, x_coord, price_x, price_y):
    x_coord_new = x_coord + dx
    dy = price_x/price_y * (x_coord - x_coord_new)
    return dy

def getHomotopy(dx, x_coord, price_x, price_y, c):
    x_coord_new = x_coord + dx
    dy_cs = price_x/price_y * (x_coord - x_coord_new)
    dy = dy_cs*(c+(1-c)*x_coord/x_coord_new)
    return dy

def getInverse(dx, x_coord, x_centre, y_centre, price_x, price_y, c):
    #x_coord_y = x_coord - x_centre
    a = (price_x/price_y) * dx + x_centre*(1-2*c)
    b = 4*(x_centre**2)*c*(1-c)
    d = 2*c
    y = (np.sqrt(a*a + b) - a)/d
    dy = x_centre - y
    return dy


#Swap functions with fee attribution logic:
def swapXOutXSide(x_out_X_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee): 
    y_in_X_side = 0
    
    if x_coord >= x_centre: # X excess curve
        y_in_X_side = getConstantSum(x_out_X_side, x_coord, x_price, y_price)
    else: #main curve
        y_in_X_side = getHomotopy(x_out_X_side, x_coord, x_price, y_price, c)
    
    y_in_X_side_fee = fee * abs(y_in_X_side)
    y_excess += y_in_X_side_fee
    x_excess += 0 
    x_coord -= abs(x_out_X_side)
    y_coord += abs(y_in_X_side)
    fee_x = y_in_X_side_fee * y_price
    
    return x_out_X_side, y_in_X_side, fee_x, x_coord, y_coord, x_excess, y_excess

def swapXOutYSide(x_out_Y_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee):
    
    y_in_Y_side = 0
    x_out_Y_side_max = x_coord - x_centre 
        
    if x_out_Y_side <= x_out_Y_side_max: #x-coordinate on the Y side
        x_out_Y_side_fee = fee * abs(x_out_Y_side) 
        if x_coord >= x_centre + x_excess: #Y main curve
            y_in_Y_side = -getInverse(x_out_Y_side, x_coord, x_centre, y_centre, x_price, y_price, c)
        else: #Y excess curve
            y_in_Y_side = getConstantSum(x_out_X_side, x_coord, x_price, y_price)
            
        x_coord -= abs(x_out_Y_side - x_out_Y_side_fee)
        y_coord += y_in_Y_side
        x_excess += x_out_Y_side_fee
        y_excess += 0
            
        return x_out_Y_side, y_in_Y_side, x_out_Y_side_fee, x_coord, y_coord, x_excess, y_excess
            
    else: #x-coordinate somewhere on the X side
        x_out_X_side = x_out_Y_side - x_out_Y_side_max 
        x_out_Y_side = x_out_Y_side_max
        x_out_Y_side_fee = fee * abs(x_out_Y_side)
    
        y_in_Y_side = getConstantSum(x_out_Y_side, x_coord, x_price, y_price)

        y_excess += 0
        x_out, y_in_X_side, y_in_X_side_fee, x_excess0, y_excess0, x_coord0, y_coord0 = swapXOutXSide(x_out_X_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
        
        x_excess += x_out_Y_side_fee
        y_excess += y_in_X_side_fee
        x_coord -= abs(x_out_Y_side - x_out_Y_side_fee)
        y_coord += (y_in_X_side+y_in_Y_side+y_excess)
        x_coord -= abs(x_out_Y_side) - abs(x_out_X_side) + x_excess

        return x_out_X_side + x_out_Y_side, y_in_Y_side + y_in_X_side, x_out_Y_side_fee + y_in_X_side_fee*y_price, x_coord, y_coord, x_excess, y_excess


def swapXInYSide(x_in_Y_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee): 
    y_out_Y_side = 0
    x_in_Y_side_fee = fee * abs(x_in_Y_side)
    x_in_Y_side -= abs(x_in_Y_side_fee)
    
    if x_coord <= x_centre + x_excess : #x-coordinate on Y excess curve
        y_out_Y_side = getConstantSum(x_in_Y_side, y_coord, x_price, y_price)
    else: # x-coordinate on Y main curve
        y_out_Y_side = -getInverse(x_in_Y_side, x_coord, x_centre, y_centre, x_price, y_price, c)
    
    x_coord += abs(x_in_Y_side)
    y_coord -= abs(y_out_Y_side) 
    x_excess += abs(x_in_Y_side_fee) 
    y_excess +=0
    
    return x_in_Y_side, y_out_Y_side, x_in_Y_side_fee, x_coord, y_coord, x_excess, y_excess

def swapXInXSide(x_in_X_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee):
    
    y_out_X_side = 0 
    x_in_X_side_max = x_centre - x_coord 
    
    if x_in_X_side <= x_in_X_side_max: #x-coordinate somewhere on the X side
        x_in_Y_side_fee = fee * abs(x_in_X_side)
        x_coord += x_in_X_side
        if x_coord <= x_centre: #X main curve
            y_out_X_side = getHomotopy(x_in_X_side, x_coord, x_price, y_price, c)
            
        else: #excess curve:
            y_out_X_side = getConstantSum(x_in_X_side, x_coord, x_price, y_price)
            
        y_coord -= y_out_X_side 
        y_out_X_side_fee = fee * abs(y_out_X_side)
        y_out_X_side -= y_out_X_side_fee
        y_excess += y_out_X_side_fee
        x_excess += 0 
        y_coord += y_out_X_side_fee

        return x_in_X_side, y_out_X_side, y_out_X_side_fee, x_coord, y_coord, x_excess, y_excess
        
    else: #x-coordinate somewhere on the Y side
        x_in_Y_side = x_in_X_side - x_in_X_side_max
        x_in_Y_side_fee = fee * abs(x_in_Y_side)
        x_in_X_side = x_in_X_side_max
        #y_out_X_side = y_coord - y_centre
        y_out_X_side = getHomotopy(x_in_X_side, x_coord, x_price, y_price, c)
        y_out_X_side_fee = fee * abs(y_out_X_side)
        y_out_X_side -= y_out_X_side_fee
        
        x_in_Y_side, y_out_Y_side, fee0, x_coord0, y_coord0, x0, y0 = swapXInYSide(x_in_Y_side, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price,  x_excess, y_excess, fee)
 
        y_coord =  y_coord -(y_out_Y_side+y_out_Y_side)
        x_coord += (x_in_Y_side+ x_in_X_side)

        y_excess += y_out_X_side_fee 
        x_excess +=x_in_Y_side_fee
         
        return x_in_Y_side+ x_in_X_side, y_out_X_side+y_out_Y_side, y_out_X_side_fee*y_price + x_in_Y_side_fee, x_coord, y_coord, x_excess, y_excess


def swapOut(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee):
    if x_coord <= x_centre: # f(x) curve
        x_out_X_side, y_in_X_side, fee, x_coord, y_coord, x_excess, y_excess = swapXOutXSide(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
        
    else: #g(y) curve
        x_out_X_side, y_in_X_side, fee, x_coord, y_coord, x_excess, y_excess = swapXOutYSide(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
        
    marginal_price = abs(x_out_X_side/y_in_X_side)
    return marginal_price, x_coord, y_coord, x_excess, y_excess, fee

def swapIn(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee):
    if x_coord <= x_centre:#f(x) curve
        x_in_X_side, y_out_X_side, fee, x_coord, y_coord, x_excess, y_excess  = swapXInXSide(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
    else: #g(y) curve
        x_in_X_side, y_out_X_side, fee, x_coord, y_coord, x_excess, y_excess  = swapXInYSide(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
    marginal_price = abs(x_in_X_side /y_out_X_side)
    return marginal_price, x_coord, y_coord, x_excess, y_excess, fee


def swap(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee):
    if amount<0:
        marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swapOut(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
    else:
        marginal_price, x_coord, y_coord, x_excess, y_excess, fee = swapIn(amount, c, x_centre, y_centre, x_coord, y_coord, x_price, y_price, x_excess, y_excess, fee)
    
    return marginal_price, x_coord, y_coord, x_excess, y_excess, fee

def new_centre(old_centre, c, p, q): 
    new_centre = old_centre * np.sqrt((1-c)/(q*(1/p) - c))
    c_new = 2 - p/q*(c+2*(1-c)*(old_centre**2/new_centre**2))
    return new_centre, c_new
    
def arbitrage_binary_search(x_coord, y_coord, price_x, price_y, target_price, c, low, high, tolerance):
    while low <= high:
        
        mid = low + (high - low)//2
        dy = getHomotopy(mid, x_coord, price_x, price_y, c)
        m_price = abs(mid/dy)
        
        if abs((m_price - target_price)/m_price) <= tolerance:
            return mid

        elif m_price < target_price: 
            low = mid + 1

        else:
            high = mid - 1

    return -1
    
def twap(prices, times, n):
    twap = np.average(prices[-n:], weights=times[:n])
    return twap
