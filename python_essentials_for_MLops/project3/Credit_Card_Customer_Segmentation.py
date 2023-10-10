import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# Configuração do logging
logging.basicConfig(filename='customer_segmentation.log', level=logging.ERROR)

# Define uma semente para reprodutibilidade
np.random.seed(42)

# Configura o estilo de plotagem
sns.set_style('whitegrid')


def load_customer_data(filename):
    """
    Carrega os dados dos clientes a partir de um arquivo CSV.
    """
    try:
        return pd.read_csv(filename)
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {str(e)}")
        return None


def explore_categorical_data(data, columns):
    """
    Exibe informações sobre colunas categóricas.
    """
    for col in columns:
        try:
            print(col)
            print(data[col].value_counts(), end='\n\n')
        except Exception as e:
            logging.error(f"Erro ao explorar dados categóricos: {str(e)}")


def plot_correlation_heatmap(data, figsize=(12, 8)):
    """
    Cria um gráfico de calor da correlação entre as variáveis numéricas.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(round(data.drop('customer_id', axis=1).corr(), 2),
                    cmap='Blues', annot=True, ax=ax)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Erro ao criar heatmap de correlação: {str(e)}")


def plot_numeric_histograms(data, figsize=(12, 10)):
    """
    Cria histogramas das variáveis numéricas.
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        data.drop('customer_id', axis=1).hist(ax=ax)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Erro ao criar histogramas: {str(e)}")


def preprocess_data(data):
    """
    Realiza o pré-processamento dos dados.
    """
    try:
        data_copy = data.copy()
        data_copy['gender'] = data['gender'].apply(
            lambda x: 1 if x == 'M' else 0)
        data_copy.replace(to_replace={'Uneducated': 0, 'High School': 1, 'College': 2,
                                      'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5}, inplace=True)
        dummies = pd.get_dummies(
            data_copy[['marital_status']], drop_first=True)
        data_copy = pd.concat([data_copy, dummies], axis=1)
        data_copy.drop(['marital_status'], axis=1, inplace=True)
        return data_copy
    except Exception as e:
        logging.error(f"Erro no pré-processamento dos dados: {str(e)}")
        return None


def scale_data(data):
    """
    Normaliza os dados usando StandardScaler.
    """
    try:
        scaler = StandardScaler()
        scaler.fit(data)
        X_scaled = scaler.transform(data)
        return pd.DataFrame(X_scaled)
    except Exception as e:
        logging.error(f"Erro na normalização dos dados: {str(e)}")
        return None


def find_optimal_clusters(data, max_clusters=10):
    """
    Encontra o número ótimo de clusters usando o método Elbow.
    """
    try:
        inertias = []
        for k in range(1, max_clusters + 1):
            model = KMeans(n_clusters=k)
            inertias.append(model.inertia_)
        return inertias
    except Exception as e:
        logging.error(f"Erro ao encontrar número ótimo de clusters: {str(e)}")
        return None


def plot_elbow_method(inertias, max_clusters=10):
    """
    Plota a inércia em função do número de clusters.
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.xticks(ticks=range(1, max_clusters + 1),
                   labels=range(1, max_clusters + 1))
        plt.title('Inertia vs Number of Clusters')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Erro ao plotar o método Elbow: {str(e)}")


def cluster_data(data, n_clusters):
    """
    Executa o K-Means com o número de clusters especificado.
    """
    try:
        model = KMeans(n_clusters=n_clusters)
        y = model.fit_predict(data)
        return y
    except Exception as e:
        logging.error(f"Erro na clusterização dos dados: {str(e)}")
        return None


def visualize_cluster_means(data, cluster_labels, numeric_columns):
    """
    Visualiza as médias de variáveis numéricas por cluster.
    """
    try:
        fig = plt.figure(figsize=(20, 20))
        for i, column in enumerate(numeric_columns):
            df_plot = data.groupby(cluster_labels)[column].mean()
            ax = fig.add_subplot(5, 2, i+1)
            ax.bar(df_plot.index, df_plot,
                   color=sns.color_palette('Set1'), alpha=0.6)
            ax.set_title(f'Average {column.title()} per Cluster', alpha=0.5)
            ax.xaxis.grid(False)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Erro ao visualizar médias por cluster: {str(e)}")


def visualize_categorical_percentages(data, cluster_labels, cat_columns):
    """
    Visualiza porcentagens de categorias categóricas por cluster.
    """
    try:
        fig = plt.figure(figsize=(18, 6))
        for i, col in enumerate(cat_columns):
            plot_df = pd.crosstab(
                index=cluster_labels, columns=data[col], values=data[col], aggfunc='size', normalize='index')
            ax = fig.add_subplot(1, 3, i+1)
            plot_df.plot.bar(stacked=True, ax=ax, alpha=0.6)
            ax.set_title(f'% {col.title()} per Cluster', alpha=0.5)
            ax.set_ylim(0, 1.4)
            ax.legend(frameon=False)
            ax.xaxis.grid(False)
            labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
            ax.set_yticklabels(labels)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(
            f"Erro ao visualizar porcentagens categóricas por cluster: {str(e)}")


def main():
    """
    Função principal para executar a análise.
    """
    try:
        # Carrega os dados dos clientes
        filename = 'customer_segmentation.csv'
        customers = load_customer_data(filename)

        if customers is None:
            return

        # Exploração inicial dos dados
        categorical_columns = ['gender', 'education_level', 'marital_status']
        explore_categorical_data(customers, categorical_columns)

        # Análise da correlação
        plot_correlation_heatmap(customers)

        # Histogramas das variáveis numéricas
        plot_numeric_histograms(customers)

        # Pré-processamento dos dados
        preprocessed_data = preprocess_data(customers)

        if preprocessed_data is None:
            return

        # Normalização dos dados
        scaled_data = scale_data(preprocessed_data)

        if scaled_data is None:
            return

        # Encontra o número ótimo de clusters
        max_clusters = 10
        inertias = find_optimal_clusters(scaled_data, max_clusters)

        if inertias is None:
            return

        # Plota o método Elbow
        plot_elbow_method(inertias, max_clusters)

        # Clusterização dos dados
        num_clusters = 6
        cluster_labels = cluster_data(scaled_data, num_clusters)

        if cluster_labels is None:
            return

        # Visualização das médias das variáveis numéricas por cluster
        numeric_columns = customers.select_dtypes(include=np.number).drop(
            ['customer_id', 'CLUSTER'], axis=1).columns
        visualize_cluster_means(customers, cluster_labels, numeric_columns)

        # Visualização das porcentagens de categorias categóricas por cluster
        cat_columns = customers.select_dtypes(include=['object'])
        visualize_categorical_percentages(
            customers, cluster_labels, cat_columns)

        # Listagem de clientes e clusters
        customer_clusters = customers[['customer_id', 'CLUSTER']]
        print(customer_clusters)
    except Exception as e:
        logging.error(f"Erro na execução do programa: {str(e)}")


if __name__ == "__main__":
    main()
