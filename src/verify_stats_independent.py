import numpy as np


def load_zfs_data(file_path: str):
    """
    Load theoretical and experimental ZFS parameters
    from a semicolon-separated CSV file.

    Columns:
        0 : no
        1 : molecule
        2 : theoretical D
        3 : theoretical E
        4 : theoretical |E/D|
        5 : experimental |D|
        6 : experimental |E|
        7 : experimental |E/D|
        8 : refs
    """

    data = np.genfromtxt(
        file_path,
        delimiter=";",
        skip_header=2,
        dtype=str,
        encoding="utf-8"
    )

    def to_float(column):
        result = []

        for value in column:
            value = value.strip()

            # Unicode minus -> normal minus
            value = value.replace("−", "-")

            try:
                result.append(float(value))
            except ValueError:
                result.append(np.nan)

        return np.asarray(result, dtype=float)

    D_calc = to_float(data[:, 2])
    E_calc = to_float(data[:, 3])
    ED_calc = to_float(data[:, 4])

    D_exp = to_float(data[:, 5])
    E_exp = to_float(data[:, 6])
    ED_exp = to_float(data[:, 7])

    return D_calc, E_calc, ED_calc, D_exp, E_exp, ED_exp


def mae(calc, exp):
    """Mean Absolute Error."""
    calc = np.asarray(calc, dtype=float)
    exp = np.asarray(exp, dtype=float)

    return np.mean(np.abs(calc - exp))


def mape(calc, exp):
    """Mean Absolute Percentage Error, %."""
    calc = np.asarray(calc, dtype=float)
    exp = np.asarray(exp, dtype=float)

    mask = exp != 0

    if not np.any(mask):
        return np.nan

    return 100 * np.mean(
        np.abs((calc[mask] - exp[mask]) / exp[mask])
    )


def rmse(calc, exp):
    """Root Mean Squared Error."""
    calc = np.asarray(calc, dtype=float)
    exp = np.asarray(exp, dtype=float)

    return np.sqrt(
        np.mean((calc - exp) ** 2)
    )


def r2(calc, exp):
    """Coefficient of determination R²."""
    calc = np.asarray(calc, dtype=float)
    exp = np.asarray(exp, dtype=float)

    ss_res = np.sum((exp - calc) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def calculate_all(calc, exp):
    """
    Calculate all metrics for one ZFS parameter.

    Returns:
        (MAE, MAPE, RMSE, R2)
    """

    return (
        mae(calc, exp),
        mape(calc, exp),
        rmse(calc, exp),
        r2(calc, exp)
    )


def calculate_zfs_metrics(
    D_calc,
    E_calc,
    ED_calc,
    D_exp,
    E_exp,
    ED_exp
):
    """
    Calculate MAE, MAPE, RMSE and R² independently for:

        |D|
        |E|
        |E/D|

    Returns
    -------
    tuple
        (
            (MAE_D,  MAPE_D,  RMSE_D,  R2_D),
            (MAE_E,  MAPE_E,  RMSE_E,  R2_E),
            (MAE_ED, MAPE_ED, RMSE_ED, R2_ED)
        )
    """

    # Experimental values are already absolute in the CSV.
    # Theoretical D and E may have signs, so take absolute values.
    D_calc = np.abs(np.asarray(D_calc, dtype=float))
    E_calc = np.abs(np.asarray(E_calc, dtype=float))
    ED_calc = np.abs(np.asarray(ED_calc, dtype=float))

    D_exp = np.asarray(D_exp, dtype=float)
    E_exp = np.asarray(E_exp, dtype=float)
    ED_exp = np.asarray(ED_exp, dtype=float)

    D_metrics = calculate_all(D_calc, D_exp)
    E_metrics = calculate_all(E_calc, E_exp)
    ED_metrics = calculate_all(ED_calc, ED_exp)

    return D_metrics, E_metrics, ED_metrics


def main():
    file_path = "data\DMRG_table_semicolon.csv"

    (
        D_calc,
        E_calc,
        ED_calc,
        D_exp,
        E_exp,
        ED_exp
    ) = load_zfs_data(file_path)

    D_metrics, E_metrics, ED_metrics = calculate_zfs_metrics(
        D_calc,
        E_calc,
        ED_calc,
        D_exp,
        E_exp,
        ED_exp
    )

    print("Metrics order: MAE, MAPE, RMSE, R²\n")

    print("|D|   :", D_metrics)
    print("|E|   :", E_metrics)
    print("|E/D| :", ED_metrics)


if __name__ == "__main__":
    main()