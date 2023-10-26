import yfinance
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning

class Ts():
    
    def __init__(self, threshold=0.05):
        """
        Klasės konstruktorius, kai nepaduodami parametrai duomenų siuntimui.
        
            Parametrai:
                threshold (float): naudojama p-reikšmė įvairiems testams, prielaidoms patikrinti
        """
        self.data = None
        self.threshold = threshold

    def __init__(self, ticker, s_date, e_date, threshold=0.05):
        """
        Klasės konstruktorius, kai paduodami parametrai duomenų siuntimui.

            Parametrai:
                ticker (str): įmonės akcijų kainos pavadinimas
                s_date (str): pradžios data
                e_date (str): pabaigos data
                threshold (float): p-reikšmė įvairiems testams, prielaidoms patikrinti
        """
        self.set_time_series(ticker, s_date, e_date)
        self.threshold = threshold

    def set_time_series(self, ticker, s_date, e_date):
        """
        Atsiunčia duomenis iš Yahoo Finance ir priskiria juos kintamajam 'data'.

            Parametrai:
                ticker (str): įmonės akcijų kainos pavadinimas
                s_date (str): pradžios data
                e_date (str): pabaigos data
        """
        data = yfinance.Ticker(ticker).history(start=s_date, end=e_date)
        data.rename(columns = {"Close": ticker}, inplace = True)
        self.data = data[ticker]

    def describe(self):
        """Spausdina aprašomąją statistiką apie laiko eilutę."""
        if self.data is not None:
            print("Trukstamų reikšmių kiekis:", self.data.isna().sum())
            print(self.data.describe())

    def plot_ts(self):
        """Braižo laiko eilutės grafiką."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data)
        plt.xlabel('Laikotarpis')
        plt.title(f"{self.data.name} laiko eilutė")
        plt.show()

    def plot_corr_funcs(self):
        """Braižo koreliacinių funkcijų ACF ir PACF grafikus laiko eilutei."""
        fig, ax = plt.subplots(2, figsize=(12,6))
        ax[0] = plot_acf(self.data, ax=ax[0], lags=20, alpha=self.threshold)
        ax[1] = plot_pacf(self.data.dropna(), ax=ax[1], lags=20, alpha=self.threshold)
        
    def differentiate(self):
        """Diferencijuoja laiko eilutę iki dviejų kartų, jeigu ji nėra stacionari pagal Augmented Dickey-Fuller (ADF) testą."""
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
        """
        Sukuria skirtingus ARIMA modelius ir išrenka tą, kurio Akaikės informacinis kriterijus AIC yra mažiausias.
        
            Parametrai:
                p (int): vėlavimų (ang. lags) kiekis
                d (int): diferencijavimų kiekis
                q (int): užvėlintų prognozuotų paklaidų kiekis
        """
        warnings.simplefilter('ignore', ValueWarning) #Ignoruoti įspėjimą, kuris pateikiamas kuriant ARIMA modelius
        best_aic = float("inf")
        best_order = None

        for p in range(0, p):
            for d in range(0, d):
                for q in range(0, q):
                    try:
                        # with warnings.catch_warnings():
                        #     warnings.
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
        """
        Nubraižo laiko eilutės prognozę, pasinaudojant geriausiu ARIMA modeliu, pasirinktam laiko tarpui.

            Parametrai:
                horizon (int): laiko intervalas, kuriam atliekama prognozė
        """
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
    """
    Patikrina ar viena laiko eilutė turi priežastinį ryšį kitai ir atvirkščiai.
    
        Parametrai:
            Ts1 (Ts): pirma lyginama laiko eilutė
            Ts2 (Ts): antra lyginama laiko eilutė
            test ('ssr_ftest'|'ssr_chi2test'|'lrtest'|'params_ftest'): pasirenkama statistika, pagal kurią apskaičiuojama p-reikšmė
            lags (int): maksimalus vėlavimų kiekis naudojamas Grangerio teste

    Galimos kintamojo 'test' reikšmės yra 'ssr_ftest', 'ssr_chi2test', 'lrtest' ir 'params_ftest'.
    """
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