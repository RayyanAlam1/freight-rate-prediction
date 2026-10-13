import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

EQUIP = ['Dry Van', 'Flatbed', 'Reefer']
GBM_COLS = ['distance', 'air_miles', 'equipment', 'weight', 'pickup_lat', 'pickup_lon',
            'delivery_lat', 'delivery_lon', 'market_index', 'dow', 'days_to_qend']


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def prep(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['weight'] = df['weight'].abs()
    df['market_index'] = df['market_index'].fillna(df.groupby('date')['market_index'].transform('median'))
    df['air_miles'] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df['equipment'] = pd.Categorical(df['equipment'], EQUIP)
    df['dow'] = df['date'].dt.dayofweek
    df['days_to_qend'] = ((df['date'] + pd.offsets.QuarterEnd(0)) - df['date']).dt.days
    df['t'] = (df['date'] - pd.Timestamp('2025-01-01')).dt.days / 365
    return df


def good_labels(df):
    rpm = df.posted_rate / df.distance
    return df[(rpm > 1.5) & (rpm < 4)]


def lin_X(df, w_med):
    w = df.weight.fillna(w_med) / 1e4
    X = pd.DataFrame({'log_dist': np.log(df.distance), 'log_dist2': np.log(df.distance) ** 2,
                      'log_mkt': np.log(df.market_index), 'w': w, 'w2': w ** 2, 't': df.t})
    ramp = np.clip(30 - df.days_to_qend, 0, None) / 30   # last 30 days of a quarter
    for eq in EQUIP:
        X['ramp_' + eq] = ramp * (df.equipment == eq)
    for eq in EQUIP[1:]:
        X[eq] = (df.equipment == eq) * 1.0
    return X


def fit(df, cols=GBM_COLS):
    y = np.log(df.posted_rate)
    w_med = df.weight.median()
    lin = LinearRegression().fit(lin_X(df, w_med), y)
    resid = y - lin.predict(lin_X(df, w_med))
    gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, random_state=42, verbose=-1,
                            deterministic=True, force_row_wise=True).fit(df[cols], resid)
    return lin, gbm, w_med, cols


def predict(model, df):
    lin, gbm, w_med, cols = model
    return np.exp(lin.predict(lin_X(df, w_med)) + gbm.predict(df[cols]))


if __name__ == '__main__':
    train = pd.read_csv('data/train_test.csv')
    val = pd.read_csv('data/validation.csv')
    model = fit(good_labels(prep(train)))

    preds = pd.Series(predict(model, prep(val)), index=val.load_id)
    sub = pd.read_csv('data/validation_predictions_template.csv')
    sub['predicted_rate'] = sub.load_id.map(preds).round(2)
    assert len(sub) == 12000 and sub.load_id.is_unique and (sub.predicted_rate > 0).all()
    sub.to_csv('validation_predictions.csv', index=False)

    # december file has no coords or market index, so take them from the data
    dec = pd.read_csv('data/december_chart_inputs.csv')
    d = dec.drop(columns='predicted_rate')
    both = pd.concat([train, val])
    for side in ['pickup', 'delivery']:
        loc = both.groupby(side)[[side + '_lat', side + '_lon']].first()
        d = d.join(loc, on=side)
    d['market_index'] = pd.to_datetime(d.date).map(val.groupby(pd.to_datetime(val.date)).market_index.median())
    dec['predicted_rate'] = predict(model, prep(d)).round(2)
    dec.to_csv('data/december_chart_inputs.csv', index=False)

    print(sub.predicted_rate.describe().round(1))
    print(dec[['date', 'predicted_rate']].to_string(index=False))
