import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_n_reports(df: pd.DataFrame, column_name: str, n=1, sort="desc", abs=False):
    """
    Returns the indices of the n rows with the highest values in the specified column.
    If abs is True, uses the absolute value of the column.
    """
    col = df[column_name].abs() if abs else df[column_name]
    if sort == "desc":
        return col.nlargest(n).index.tolist()
    elif sort == "asc":
        return col.nsmallest(n).index.tolist()
    else:
        raise ValueError("sort must be either 'desc' or 'asc'")


def find_reports_based_on_criteria(df: pd.DataFrame, thresholds: dict[str, float], n=1):
    """
    Returns the indices of the n rows where all specified columns are >= their threshold values.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col, threshold in thresholds.items():
        mask &= df[col] >= threshold
    filtered = df[mask]
    return filtered.index[:n].tolist()


def print_reports(df: pd.DataFrame, indices: list[int], metric: str):
    """
    Prints the n reports.
    """
    print(f"metric: {metric}, n: {len(indices)}")
    for index in indices:
        print("---------------------------------------------------------")
        print(f"Score: {df.iloc[index][metric]}, Index: {index}")
        print(f"{df.iloc[index]["predicted"]}")
        print("--------")
        print(f"{df.iloc[index]["target"]}")
        print("\n\n")


def column_average_score(df, column_name):
    if pd.api.types.is_numeric_dtype(df[column_name]):
        return df[column_name].mean()
    else:
        raise ValueError(f"Column {column_name} is not numeric.")


def column_median_score(df, column_name):
    if pd.api.types.is_numeric_dtype(df[column_name]):
        return df[column_name].median()
    else:
        raise ValueError(f"Column {column_name} is not numeric.")


def plot_column_distribution(df: pd.DataFrame, column_name: str, bins=10):
    if pd.api.types.is_numeric_dtype(df[column_name]):
        plt.figure(figsize=(6, 4))
        plt.hist(df[column_name], bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    else:
        raise ValueError(f"Column {column_name} is not numeric.")


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Plots a correlation matrix for the DataFrame.
    """
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()
