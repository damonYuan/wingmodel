import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import math
import urllib.request
import json
import re
from scipy import optimize

V_SF = 0.45
DOMAIN = 'https://www.okx.com'


def get_strike(row):
    pattern = re.escape(f"{row['uly']}") + r"-(\d{6})-(\d+)-."
    match = re.search(pattern, row['instId'])
    return float(match.group(2))


def get_expiry(row):
    pattern = re.escape(f"{row['uly']}") + r"-(\d{6})-(\d+)-."
    match = re.search(pattern, row['instId'])
    # return datetime.datetime.strptime(match.group(1), '%y%m%d').date()
    return match.group(1)


def get_option_market_data(uly, expiry=None):
    url = f'{DOMAIN}/api/v5/public/opt-summary?uly={uly}'
    print(url)
    response = urllib.request.urlopen(url)
    res = response.read().decode('utf-8')
    data = json.loads(res)['data']
    df = etl(pd.read_json(json.dumps(data)))
    if expiry is None:
        return df
    else:
        return df[df['expiry'] == expiry]


# extract, transform and load
def etl(data_df):
    data_df.loc[:, 'deltaBS'] = data_df.loc[:, 'deltaBS'].astype(float)
    data_df.loc[:, 'askVol'] = data_df.loc[:, 'askVol'].astype(float)
    data_df.loc[:, 'bidVol'] = data_df.loc[:, 'bidVol'].astype(float)
    data_df.loc[:, 'fwdPx'] = data_df.loc[:, 'fwdPx'].astype(float)
    data_df.loc[:, 'strike'] = data_df.apply(get_strike, axis=1)
    data_df.loc[:, 'expiry'] = data_df.apply(get_expiry, axis=1)
    data_df.loc[:, 'moneyness'] = data_df['strike'] / data_df['fwdPx']
    data_df.loc[:, 'mid_vol'] = (data_df['askVol'] + data_df['bidVol']) / 2
    data_df = data_df.sort_values('moneyness')
    return data_df[(data_df['askVol'] != 0) & (data_df['bidVol'] != 0)]


def segment(x, k, a1, b1, c1, a2, b2):
    c2 = a1 + b1 + c1 - a2 - b2
    return np.piecewise(x, [x <= k, x > k], [lambda x: c1 + b1 * x + a1 * x * x, lambda x: c2 + b2 * x + a2 * x * x])


def solution(y, a, b, c, left=True):
    d = (b**2) - (4*a*(c-y))
    l = (-b - math.sqrt(d)) / (2 * a)
    r = (-b + math.sqrt(d)) / (2 * a)
    if left:
        return l
    else:
        return r


def get_closest(value, arr):
    index = (np.abs(arr - value)).argmin()
    return arr[index]


def draw(df):
    x = df['moneyness'].to_numpy()
    # print(x)
    # y = np.piecewise(x, [x <= 1, x > 1], [lambda x: 1 - x + 0.5 * x * x, lambda x: 0.6 - 0.2 * x + 0.1 * x * x]) + np.random.normal(0, 1, n) * 0.01
    y = df['mid_vol'].to_numpy()
    # print(y)
    data = pd.DataFrame({'x': x, 'y': y})

    # # Fit models
    p0 = [1, 1, 1, 1, 1, 1]
    popt, pcov = optimize.curve_fit(segment, x, y, p0=p0)
    print(popt)

    k, a1, b1, c1, a2, b2 = popt
    c2 = a1 + b1 + c1 - a2 - b2

    # another way to do the fitting but less smooth compare with curve_fit
    # model1 = ols('y ~ 1 + x + I(x**2)', data=data[data['x'] <= 1]).fit()
    # model2 = ols('y ~ 1 + x + I(x**2)', data=data[data['x'] > 1]).fit()
    #
    # # Get coefficients
    # c1, b1, a1 = model1.params
    # c2, b2, a2 = model2.params

    # upper limit
    CUT_DN = get_closest(solution(V_SF, a1, b1, c1, True), x)
    CUT_DN_Y = segment(CUT_DN, k, a1, b1, c1, a2, b2)
    CUT_UP = get_closest(solution(V_SF, a2, b2, c2, False), x)
    CUT_UP_Y = segment(CUT_UP, k, a1, b1, c1, a2, b2)
    print(f"CUT_DN: {CUT_DN}; CUT_UP: {CUT_UP}")

    filtered_x = x[(x >= CUT_DN) & (x <= CUT_UP)]
    # Plot data and curve
    plt.plot(x, y, 'o', label='orders', markersize=2)
    plt.plot(filtered_x, [segment(z, k, a1, b1, c1, a2, b2) for z in filtered_x],
             label='curve', color='r')
    plt.legend()
    plt.xlabel('K/S')
    plt.ylabel('vol')
    plt.axvline(x=1, color='g', linestyle='--')
    plt.axvline(x=CUT_DN, color='g', linestyle='--')
    plt.axvline(x=CUT_UP, color='g', linestyle='--')
    plt.axhline(y=V_SF, color='b', linestyle='--')
    tick_labels = ['CUT_DN', 'CUT_UP', '1']
    tick_positions = [CUT_DN, CUT_UP, 1]
    plt.xticks(tick_positions, tick_labels)
    plt.plot([np.min(x), CUT_DN], [CUT_DN_Y, CUT_DN_Y], color='r', linestyle='solid')
    plt.plot([CUT_UP, np.max(x)], [CUT_UP_Y, CUT_UP_Y], color='r', linestyle='solid')
    plt.show()


if __name__ == '__main__':
    df = etl(get_option_market_data('BTC-USD'))
    print(f"all expiry: {df['expiry'].sort_values().unique()}")
    print(f"all strikes: {df['strike'].sort_values().unique()}")
    expiry = '230623'
    df = df[df['expiry'] == expiry]
    print(f"fwdPx for {expiry}: {df.loc[:, 'fwdPx'].sort_values().unique()}")
    print(f"size: {df.size}")
    draw(df)


