### MAE — Mean Absolute Error

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
y_i^{\mathrm{calc}} - y_i^{\mathrm{exp}}
\right|
$$

### MAPE — Mean Absolute Percentage Error

$$
\mathrm{MAPE}
=
\frac{100\%}{N}
\sum_{i=1}^{N}
\left|
\frac{
y_i^{\mathrm{calc}} - y_i^{\mathrm{exp}}
}{
y_i^{\mathrm{exp}}
}
\right|
$$

### RMSE — Root Mean Squared Error

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
y_i^{\mathrm{calc}} - y_i^{\mathrm{exp}}
\right)^2
}
$$

### R² — Coefficient of Determination

$$
R^2
=
1 -
\frac{
\sum_{i=1}^{N}
\left(
y_i^{\mathrm{exp}} - y_i^{\mathrm{calc}}
\right)^2
}{
\sum_{i=1}^{N}
\left(
y_i^{\mathrm{exp}} - \overline{y}^{\mathrm{exp}}
\right)^2
}
$$

where

$$
\overline{y}^{\mathrm{exp}}
=
\frac{1}{N}
\sum_{i=1}^{N}
y_i^{\mathrm{exp}}
$$

For ZFS analysis, these metrics can be calculated independently for:

$$
|D|,\qquad |E|,\qquad \left|\frac{E}{D}\right|
$$