#BlackScholes.py
#module implementing the Black-Scholes Formula

import math

# TODO: import ChrustSolver
from scipy.stats import distributions

from ..price import Price

N = distributions.norm.cdf #dystrybuanta
fi = distributions.norm.pdf #rozklad prawdopodobienstwa

def BSCall(S: Price, K: Price, r: float, volatility: float, Time: float, d: float =0.0):
  #delta in arguments is small delta - the dividend yield
  # TODO: print(f"BSCall, S: {S}, K: {K}, r: {r}, volatility: {volatility}, Time: {Time}")
  # on expiration date Time=0, so the option value is S-K
  if Time == 0:
    callPrice = max(Price(0), S - K)
    intrinsicValue = callPrice
    delta = 1 if S > K else 0
    gamma = 0
    theta = 0
    vega = 0
    rho = 0
  else:
    d1 = (math.log(float(S)/float(K)) + ((r-d+(volatility**2)/2) * Time))/(volatility * math.sqrt(Time))
    d2 = d1 - volatility * math.sqrt(Time)
    #print "d1, d2:", d1, d2
    delta = N(d1)
    gamma = fi(d1) / (float(S)*volatility*math.sqrt(Time))
    theta = - (float(S) * fi(d1) * volatility)/ (2*math.sqrt(Time)) - r * float(K) * math.exp(-r*Time)*N(d2) #lack of dividend-related factor
    vega = float(S) * fi(d1) * math.sqrt(Time)
    rho = float(K) * Time * math.exp(-r*Time)*N(d2)
    callPrice = float(S)*distributions.norm.cdf(d1) - math.exp(-r*Time)*float(K)*distributions.norm.cdf(d2)
    intrinsicValue = max(0, float(S)-float(K)*math.exp(-r*Time))
  timeValue = callPrice - intrinsicValue
  # print({"price": callPrice, "delta":delta, "gamma":gamma, "theta": theta, "vega": vega, "rho": rho, \
  #   "intrinsicValue": intrinsicValue, "timeValue": timeValue})
  #return callPrice  #maybe return a tuple (or dict) with the price and all the Greeks?
  return {"price": callPrice, "delta":delta, "gamma":gamma, "theta": theta, "vega": vega, "rho": rho, \
    "intrinsicValue": intrinsicValue, "timeValue": timeValue}
  

def BSPut(S: Price, K: Price, r: float, volatility: float, Time: float, d: float=0.0):
  # TODO: print(f"BSPut, S: {S}, K: {K}, r: {r}, volatility: {volatility}, Time: {Time}")
  if Time == 0:
    putPrice = max(Price(0), K-S)
    intrinsicValue = putPrice
    delta = 1 if S < K else 0
    gamma = 0
    theta = 0
    vega = 0
    rho = 0
  else:
    d1 = (math.log(float(S)/float(K)) + ((r-d+(volatility**2)/2) * Time))/(volatility * math.sqrt(Time))
    d2 = d1 - volatility * math.sqrt(Time)
    #print "d1, d2:", d1, d2
    delta = N(d1) - 1
    gamma = fi(d1) / (float(S) * volatility * math.sqrt(Time))
    theta = -(float(S)*fi(d1)*volatility)/(2*math.sqrt(Time)) + r*float(K)*math.exp(-r*Time)*N(-d2) #as above - dividend-related factor should be added
    vega = float(S)* fi(d1) * math.sqrt(Time)
    rho = - float(K) * Time * math.exp(-r*Time) * N(-d2)
    putPrice = - float(S)*distributions.norm.cdf(-d1) + math.exp(-r*Time)*float(K)*distributions.norm.cdf(-d2)
    intrinsicValue = max(0, float(K)*math.exp(-r*Time)-float(S))
  timeValue = putPrice - intrinsicValue
  return {"price": putPrice, "delta":delta, "gamma":gamma, "theta": theta, "vega": vega, "rho": rho, \
    "intrinsicValue": intrinsicValue, "timeValue": timeValue}
  # return putPrice
  

#function calculating the volatility implied in the   prices of options
def impliedVolatilityInCall(C, S, K, Time, r, d=0.0):
  #if C > S - math.exp(r*Time) * K:
    #bullshit: raise RuntimeError("Cannot calculate implied volatility! - There is opportunity of arbitrage by writing a call, buying stock for borrowed money!")
  #if C < S - math.exp(r*Time) * K:
    #bullshit either:raise RuntimeError("Here is arbitrage opportunity: short sell stocks, buy calls, "
  if C > S:
    print("Impossible C, S, K, Time, r, d: ", C, S, K, Time, r, d)
    raise RuntimeError("Impossible! (arbitrage opportunity - call is more expensive than spot!)")
  #if C < math.exp(r*Time) * S - K: #that was wrong
  if C < S - K * math.exp(-r*Time):
    print("Impossible C, S, K, Time, r, d: ", C, S, K, Time, r, d)
    raise RuntimeError("Impossible! (arbitrage opportunity - call to cheap! call cheaper than intrinsic option value.)")
    
  #a simple artefact function
  # TODO
  # def innerCall(vega):
  #   return BSCall(S, K, r, vega, Time, d)
  #
  # return ChrustSolver.solveNonlinear(C, innerCall, 0.0005, (0.001, 50))
    
def impliedVolatilityInPut(P, S, K, Time, r, d=0.0):
  if P >= math.exp(r*Time) * K:
    print("Impossible P, S, K, Time, r, d: ", P, S, K, Time, r, d)
    raise RuntimeError("Impossible! (arbitrage opportunity - put too expensive!)")
  if P < math.exp(-r*Time)*K - S:
    print("Impossible P, S, K, Time, r, d: ", P, S, K, Time, r, d)
    raise RuntimeError("Impossible! (arbitrage opportunity - put too cheap!)")
    
  #simple artefact function:
  def innerPut(vega):
    return BSPut(S, K, r, vega, Time, d)
  # TODO
  # return ChrustSolver.solveNonlinear(P, innerPut, 0.0005, (0.001, 50))
  

#function calculates whether there is an arbitrage opportunity (and points it) on the futures and options market
#we should include the bid-ask spread here, ey?  - the C, P, Futures and maybe even r should passed as tuples
def findOptionArbitrage(C, P, K, Futures, Time, r):
  if isinstance(r, float):
    FVoptionSpread = ((C[1]-P[0]) * math.exp(r*Time), (C[0]-P[1])*math.exp(r*Time))
  elif isinstance(r, tuple) and len(r) == 2:
    FVoptionSpread = ((C[1]-P[0]) * math.exp(r[0]*Time), (C[0]-P[1])*math.exp(r[1]*Time))
  else:
    raise ValueError("Invalid interest rate format! It should be float or two-element tuple, is %s." % r.__class__)
  if FVoptionSpread[0] <= Futures[1] - K and FVoptionSpread[1] >= Futures[0] - K: #check it!
    return False #no arbitrage opportunity
  else:
    if FVoptionSpread[0] > Futures[1] - K: #ckeck it
      print("Arbirage opportunity: buy Put, sell Call and take long Forward position.")
    else:
      print("Arbitrage opportunity: buy Call, sell Put and take short Forward position.")
    return True #arbitrage opportunity as printed above
  
def findOptionArbitrage2(C1, P1, K1, C2, P2, K2, r, T):
  if (C1[1] - P1[0]) - (C2[0] - P2[1]) < (K1 - K2)*math.exp(-r*T): #comparing synthetic long position on K1 with synthetic short on K2
    print("Arbitrage opportunity: create synthetic long on %f and synthetic short on %f expiring %f." % (K1, K2, T))
    return True
  if (C1[0] - P1[1]) - (C2[1] - P2[0]) > (K2 - K1)*math.exp(-r*T): #check it! - should be ok, as it is a mirror from the above equation
    print("Arbitrage opportunity: create synthetic short on %f and synthethic long on %f (both expiring %f)." % (K1, K2, T))
    return True
  return False
