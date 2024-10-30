#install BigVAR, AER, quantmod, PerformanceAnalytics
library(BigVAR)
library(AER)
library(quantmod)
library(PerformanceAnalytics)

rm(list=ls())


data(Y)
mod1 <- constructModel(Y, p=4, struct="Basic", gran=c(150,10), h=1, cv="Rolling", verbose=FALSE, IC=TRUE, model.controls=list(intercept=TRUE))
results = cv.BigVAR(mod1)
results
plot(results)
SparsityPlot.BigVAR.results(results)
predict(results,n.ahead=1)
predict(results,n.ahead=1, confint=TRUE)
coef(results)


----
  
dfAAPL = getSymbols("AAPL", from="2020-01-01", to="2021-12-31", auto.assign=F)
rAAPL = as.numeric(coredata(CalculateReturns(dfAAPL[,"AAPL.Adjusted"])[-c(1),]))
dfSP500 = getSymbols("^GSPC", from="2020-01-01", to="2021-12-31", auto.assign=F)
rSP500 = as.numeric(coredata(CalculateReturns(dfSP500[,"GSPC.Adjusted"])[-c(1),]))
Y2 <- cbind(AAPL = rAAPL, SP500 = rSP500)

modfin <- constructModel(Y2, p=2, struct="Basic", gran=c(150,10), h=1, cv="Rolling", verbose=FALSE, IC=TRUE, model.controls=list(intercept=TRUE))
resultsfin = cv.BigVAR(modfin)
coef(resultsfin)
SparsityPlot.BigVAR.results(resultsfin)
