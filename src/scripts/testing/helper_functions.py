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

# TODO

def find_and_print_n_reports(df: pd.DataFrame, column_name: str, n=1, sort="desc", abs=False):
    """
    Finds and prints the n reports based on the specified column and other criteria.
    """
    row_indices = find_n_reports(df, column_name, n=n, sort=sort, abs=abs)
    print(f"metric: {column_name}, n: {n}")
    for index in row_indices:
        print("---------------------------------------------------------")
        print(f"Score: {df.iloc[index][column_name]}, Index: {index}")
        print(f"{df.iloc[index]["predicted"]}")
        print("--------")
        print(f"{df.iloc[index]["target"]}")
        print("\n\n")
    return row_indices


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
        plt.figure(figsize=(10, 6))
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
