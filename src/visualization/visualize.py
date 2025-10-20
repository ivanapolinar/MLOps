import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_target_distribution(df, figures_dir):
    plt.figure(figsize=(7, 5))
    load_counts = df["Load_Type"].value_counts()
    sns.barplot(x=load_counts.index, y=load_counts.values, edgecolor="black")
    plt.title('Distribución del Target (Load_Type)')
    plt.xlabel('Tipo de carga')
    plt.ylabel('Cantidad de registros')
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(load_counts.values):
        plt.text(
            i,
            v + (0.01 * max(load_counts.values)),
            f"{v}",
            ha="center",
            fontsize=10
        )
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/target_distribution.png")
    plt.close()


def plot_histograms_boxplots(df, figures_dir, focus_cols):
    for col in focus_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Histograma
        axes[0].hist(
            df[col].dropna(),
            bins=60,
            alpha=0.75,
            color='skyblue',
            edgecolor='black'
        )
        axes[0].set_title(f"Histograma: {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frecuencia")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        # Boxplot
        axes[1].boxplot(
            df[col].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightcoral', color='black'),
            medianprops=dict(color='black')
        )
        axes[1].set_title(f"Boxplot: {col}")
        axes[1].set_ylabel(col)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/hist_box_{col}.png")
        plt.close()


def calculate_outliers(df, num_cols, figures_dir):
    outlier_info = {}
    outlier_stats = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5*IQR
        lim_sup = Q3 + 1.5*IQR
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Limite inferior': lim_inf,
            'Limite superior': lim_sup
        }
        outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)]
        pct = (outliers.shape[0] / df.shape[0]) * 100
        outlier_stats.append({'Variable': col, 'Porcentaje_Outliers': pct})
    pd.DataFrame(outlier_stats).sort_values(
        by='Porcentaje_Outliers',
        ascending=False
    ).to_csv(
        path_or_buf=f"{figures_dir}/outlier_stats.csv",
        index=False
    )
    return outlier_info, outlier_stats


def ts_plot(df, col, rule, figures_dir):
    s = df.set_index("date")[col].resample(rule).mean()
    s_smooth = s.rolling(window=10, center=True).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(s, label='Promedio', alpha=0.6)
    plt.plot(
        s_smooth,
        color='orange',
        linewidth=2,
        label='Tendencia suavizada'
    )
    plt.title(f"{col} promedio por {rule}")
    plt.xlabel("Fecha")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/trend_{col}_{rule}.png")
    plt.close()


def bar_by_category(df, value_col, cat_col, figures_dir, agg="mean"):
    tab = (df.groupby(cat_col, dropna=False)[value_col]
           .agg(agg)
           .sort_values(ascending=False))
    plt.figure()
    tab.plot(
        kind="bar",
        title=f"{value_col} por {cat_col} ({agg})"
    )
    plt.ylabel(value_col)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/bar_{value_col}_{cat_col}.png")
    plt.close()
    return tab


def plot_correlation_matrix(df, num_cols, figures_dir):
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(
        label="Matriz de Correlación (variables numéricas)",
        fontsize=13,
        pad=10
    )
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/correlation_matrix.png")
    plt.close()


def plot_scatter_pairs(df, pairs, figures_dir):
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            sample = df[[x, y]].dropna()
            if len(sample) > 20000:
                sample = sample.sample(20000, random_state=42)
            plt.figure(figsize=(6, 4))
            sns.scatterplot(
                data=sample,
                x=x,
                y=y,
                alpha=0.5,
                edgecolor=None,
                s=20
            )
            sns.regplot(
                data=sample,
                x=x,
                y=y,
                scatter=False,
                color='red',
                ci=None,
                line_kws={'lw': 1}
            )
            plt.title(f"{y} vs {x}", fontsize=12)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/scatter_{y}_vs_{x}.png")
            plt.close()


def main(
        input_path="data/clean/steel_energy_clean.csv",
        figures_dir="reports/figures"
):
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    num_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    # cat_cols = df.select_dtypes(
    #     include=["object", "category"]
    # ).columns.tolist()

    # Distribución del target
    plot_target_distribution(df, figures_dir)

    # Histogramas y boxplots
    focus_cols = [c for c in [
        "Usage_kWh", "CO2(tCO2)", "NSM",
        "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"
    ] if c in df.columns]
    plot_histograms_boxplots(df, figures_dir, focus_cols)

    # Outliers
    calculate_outliers(df, num_cols, figures_dir)

    # Tendencias temporales
    if "Usage_kWh" in df.columns:
        for rule in ["15min", "h", "D", "W"]:
            ts_plot(df, "Usage_kWh", rule, figures_dir)

    # Comparativas por categoría
    if "Usage_kWh" in df.columns:
        for cat in ["WeekStatus", "Day_of_week", "Load_Type"]:
            if cat in df.columns:
                bar_by_category(
                    df,
                    "Usage_kWh",
                    cat,
                    figures_dir,
                    agg="mean"
                )

    # Correlación
    plot_correlation_matrix(df, num_cols, figures_dir)

    # Dispersión de pares relevantes
    pairs = [
        ("Lagging_Current_Reactive.Power_kVarh", "Usage_kWh"),
        ("Leading_Current_Reactive_Power_kVarh", "Usage_kWh"),
        ("Lagging_Current_Power_Factor", "Usage_kWh"),
        ("Leading_Current_Power_Factor", "Usage_kWh"),
        ("Usage_kWh", "CO2(tCO2)")
    ]
    plot_scatter_pairs(df, pairs, figures_dir)


if __name__ == "__main__":
    main()
