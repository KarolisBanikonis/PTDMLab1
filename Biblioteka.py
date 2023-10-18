import yfinance
import matplotlib.pyplot as plt
import pandas as pd
import sys
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA

class Ts():
    
    def __init__(self, threshold=0.05):
        self.data = None
        self.threshold = threshold

    def __init__(self, ticker, s_date, e_date, threshold=0.05):
        self.set_time_series(ticker, s_date, e_date)
        self.threshold = threshold

    def set_time_series(self, ticker, s_date, e_date):
        data = yfinance.Ticker(ticker).history(start=s_date, end=e_date)
        data.rename(columns = {"Close": ticker}, inplace = True)
        self.data = data[ticker]

    def describe(self):
        if self.data is not None:
            print("Trukstamų reikšmių kiekis:", self.data.isna().sum())
            print(self.data.describe())

    def plot_ts(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data)
        plt.xlabel('Laikotarpis')
        plt.title(f"{self.data.name} laiko eilutė")
        plt.show()

    def plot_corr_funcs(self):
        fig, ax = plt.subplots(2, figsize=(12,6))
        ax[0] = plot_acf(self.data, ax=ax[0], lags=20, alpha=self.threshold)
        ax[1] = plot_pacf(self.data.dropna(), ax=ax[1], lags=20, alpha=self.threshold)
        
    def differentiate(self):
        i = 0
        while i < 2:
            i += 1
            stationary = adfuller(self.data)
            if stationary[1] > self.threshold:
                print(f'{self.data.name} laiko eilutė nėra stacionari, atliekamas diferencijavimas {i} kartą.')
                self.data = self.data - self.data.shift(1)
                self.data = self.data.iloc[1:]
            else:
                print(f'{self.data.name} laiko eilutė yra stacionari.')
                break

    def grid_select_arima(self, p, d, q):
        best_aic = float("inf")
        best_order = None

        for p in range(0, p):
            for d in range(0, d):
                for q in range(0, q):
                    try:
                        model = ARIMA(self.data, order = (p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                    except:
                        continue

        self.arima_model = ARIMA(self.data, order = best_order)
        self.arima_fit = self.arima_model.fit()
        print(self.arima_fit.summary())

    def arima_plot_forecast(self, horizon):
        if horizon >= len(self.data):
            print("Prognozuojamas dienų kiekis privalo būti mažesnis, nei duomenų kiekis!")
            sys.exit(2)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(self.data, label = f"{self.data.name} laiko eilutė")
        plot_predict(self.arima_fit, self.data.index[-horizon], self.data.index[-1], ax=ax, alpha=self.threshold)
        plt.title(f'ARIMA {self.arima_model.order} modelio prognozė')
        plt.xlabel('Data')
        plt.legend()
        plt.show()

def granger_Ts(Ts1, Ts2, test = 'ssr_ftest', lags = 10):
    if Ts1.threshold != Ts2.threshold:
        print("Lyginamų laiko eilučių objektų threshold reikšmės privalo sutapti!")
        sys.exit(1)
    df = pd.concat([Ts1.data, Ts2.data], axis=1)
    name1 = Ts2.data.name
    name2 = Ts1.data.name
    granger1 = grangercausalitytests(df[[name2, name1]], maxlag = lags, verbose=False)
    granger2 = grangercausalitytests(df[[name1, name2]], maxlag = lags, verbose=False)

    def print_granger_results(granger, name1, name2):
        for key in granger.keys():
            try:
                v = granger[key][0][test]
            except KeyError as ke:
                print('Nurodyta neteisinga test parametro reikšmė –', ke)
            if v[1] > Ts1.threshold:
                print(f"Vėlavimas = {key}, p-reikšmė = {round(v[1], 2)}, {name1} nedaro įtakos {name2}.")
            else:
                print(f"Vėlavimas = {key}, p-reikšmė = {round(v[1], 2)}, {name1} daro įtaką {name2}.")

    print_granger_results(granger1, name1, name2)
    print_granger_results(granger2, name2, name1)