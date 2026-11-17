# Data dictionary

Two layers: the published UCI columns (as delivered) and the derived columns the
pipeline adds. Names are exact.

## 1. Raw columns (`hour.csv`)

| Column | Type | Description | Pipeline use |
| --- | --- | --- | --- |
| `instant` | int | Row index in the original file | Dropped |
| `dteday` | date | Calendar day | Used to build `timestamp`, then dropped |
| `season` | int | 1 = Jan–Mar, 2 = Apr–Jun, 3 = Jul–Sep, 4 = Oct–Dec | Rebuilt from month, kept as feature |
| `yr` | int | 0 = 2011, 1 = 2012 | Rebuilt from date, kept as feature |
| `mnth` | int | Month 1–12 | Rebuilt, kept |
| `hr` | int | Hour 0–23 | Rebuilt, kept |
| `holiday` | int | 1 if the day is a US federal holiday | Kept as feature |
| `weekday` | int | Day of week, 0 = Sunday in the source file | Rebuilt as pandas 0 = Monday, kept |
| `workingday` | int | 1 if neither weekend nor holiday | Kept as feature |
| `weathersit` | int | 1 clear, 2 mist/cloud, 3 light rain/snow, 4 heavy rain/snow | Kept as ordinal feature |
| `temp` | float | Normalised temperature (°C / 41) | Kept as feature |
| `atemp` | float | Normalised apparent temperature (°C / 50) | Kept as feature |
| `hum` | float | Normalised humidity (0–1) | Kept as feature |
| `windspeed` | float | Normalised wind speed (/ 67) | Kept as feature |
| `casual` | int | Rentals by casual users | **Dropped** — component of the target |
| `registered` | int | Rentals by registered users | **Dropped** — component of the target |
| `cnt` | int | Total rentals for the hour | Target |

`casual + registered == cnt` for every row; keeping either component would leak
the label.

## 2. Derived columns (clean stage)

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime64[ns] | `dteday` + `hr`; the index of the hourly grid |
| `observed` | int (0/1) | 0 for the 165 grid hours absent from the published file |
| `day_of_year` | int | 1–366, from `timestamp` |
| `is_weekend` | int | 1 when `weekday >= 5` |
| `weather_label` | string | `Clear`, `Mist / cloud`, `Light rain / snow`, `Heavy rain / snow` |

Gap rows carry forward-filled weather and a `NaN` target, and are excluded from
modelling.

## 3. Derived columns (feature stage)

### Cyclical encodings

All bounded in [−1, 1].

| Column | Definition |
| --- | --- |
| `hour_sin`, `hour_cos` | `sin/cos(2π·hr/24)` |
| `hour_sin2`, `hour_cos2` | second harmonic, `sin/cos(4π·hr/24)` |
| `hour_sin3`, `hour_cos3` | third harmonic, `sin/cos(6π·hr/24)` |
| `weekday_sin`, `weekday_cos` | `sin/cos(2π·weekday/7)` |
| `month_sin`, `month_cos` | `sin/cos(2π·(mnth−1)/12)` |
| `doy_sin`, `doy_cos` | `sin/cos(2π·day_of_year/365.25)` |

The three hour harmonics let a linear model represent the twin commute peaks;
gradient boosting simply ignores the redundancy.

### Demand history

| Column | Definition |
| --- | --- |
| `demand_lag_1h` | `log1p(cnt)` one hour earlier |
| `demand_lag_24h` | `log1p(cnt)` 24 hours earlier |
| `demand_lag_168h` | `log1p(cnt)` 168 hours (one week) earlier |
| `demand_roll_24h` | Mean of `log1p(cnt)` over the previous 24 hours |
| `demand_roll_168h` | Mean of `log1p(cnt)` over the previous 168 hours |

Rolling windows use `min_periods = 0.8 × window` so a single gap hour does not
wipe out the feature. The seasonal-naive baseline is
`expm1(demand_lag_168h)`, back on the rental scale.

## 4. Output columns (`test_predictions.csv`)

| Column | Description |
| --- | --- |
| `timestamp` | Hour being forecast |
| `hour`, `weekday`, `workingday` | Calendar context for the error breakdown |
| `weather_label` | Weather category (the ordinal `weathersit` is only in the modelling frame) |
| `y_true` | Actual rentals |
| `y_pred` | Predicted rentals (inverted from `log1p`) |
| `residual` | `y_pred − y_true` (negative means under-prediction) |
