import os
import json

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans


def process_json_files():
    read_path = r'C:\Users\samuello\Downloads\III\宏英\code\data\DATA\1\RawData'
    write_path = r'C:\Users\samuello\Downloads\III\宏英\code\data\DATA\1\data'
    files = os.listdir(read_path)
    for file in files:
        logger.info('Proccessing {} ...'.format(file))
        with open(os.path.join(read_path, file), 'r', encoding='utf-8') as rf:
            write_filename = file[:-5] + '.json'
            with open(os.path.join(write_path, write_filename), 'w') as wf:
                lines = rf.readlines()
                for line in lines:
                    try:
                        line = line[:-2]
                        assert json.loads(line)
                        line = json.loads(line)
                        new_keys = {
                            "材料溫度": "M_tem",
                            "環境溫度": "A_tem",
                            "環境濕度": "A_hum",
                            "thickness": "final thickness",
                            "材料偏擺度": "M_yaw"
                        }
                        for old_key, new_key in new_keys.items():
                            line[new_key] = line.pop(old_key)
                        line = json.dumps(line)
                        # print(line)
                        # exit(0)
                        wf.write(line)
                        wf.write('\n')
                    except Exception as e:
                        pass
                # exit(0)

def precess_groupby_data():
    path = r'C:\Users\samuello\Downloads\III\旺欉\data\20220705-0706'
    file_list = os.listdir(path)

    cols = ['Timestamp', 'ingot', 'discharge', 'mould', 'oil_pressure', 'bucket', 'E_temperature', 'E_humidity']
    df = pd.DataFrame(columns=cols)
    data = pd.DataFrame()
    i = 1
    for file in file_list:
        logger.info('Precessing no.{} file: {} ...'.format(i, file))

        try:
            data = pd.read_json(os.path.join(path, file), lines=True)
            data = data.drop(columns=['pie'])
            data = data.groupby(by=['Timestamp'], as_index=False).mean()
            data = data[cols]
            # assert (data.columns == ['Timestamp', 'ingot', 'discharge', 'mould', 'oil_pressure', 'bucket']).all()
            df = pd.concat([df, data], axis=0)
            i = i + 1
        except Exception as e:
            print(data.head(10))
            print('Error: ', e)
            exit(0)

    df = df.reset_index(drop=True)
    df.to_csv('Jul-5-6-data.csv', index=False)


def plot_by_pie():
    read_path = r'C:\Users\samuello\Downloads\III\旺欉\data\202201'
    write_path = r'C:\Users\samuello\Downloads\III\旺欉\graph\202201'
    files = os.listdir(read_path)
    for file in files:
        df = pd.read_json(os.path.join(read_path, file), lines=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

        current_directory = '{}\\{}'.format(write_path, file[:-5])
        try:
            os.mkdir(current_directory)
        except Exception as e:
            logger.info(e)

        df_pie_timestamp = df.loc[df['pie'] == 999, 'Timestamp']

        for col in df.columns:
            if col not in ['Timestamp', 'pie']:
                for i in range(len(df_pie_timestamp) - 1):
                    # df_ = df[
                    #     (df['Timestamp'] >= df['Timestamp'].iloc[0] + timedelta(minutes=10) * i) &
                    #     (df['Timestamp'] < df['Timestamp'].iloc[0] + timedelta(minutes=10) * (i + 1))
                    # ]
                    # df = df[df['Timestamp'] <= pd.to_datetime('2022-01-06 10:10:59', format='%Y-%m-%d %H:%M:%S')]
                    # print('15分鐘內出現pie=999的次數: ', len(df[df['pie'] == 999]))

                    next_directory = '{}\\{}'.format(current_directory, str(i))
                    try:
                        if not os.path.isdir(next_directory):
                            os.mkdir(next_directory)
                    except Exception as e:
                        logger.info(e)

                    df_ = df[
                        (df['Timestamp'] >= df_pie_timestamp.iloc[i]) &
                        (df['Timestamp'] < df_pie_timestamp.iloc[i + 1])
                        ]

                    _, ax = plt.subplots()
                    plt.plot(df_.index, df_[col], label=col)
                    # myFmt = mdates.DateFormatter('%H:%M')
                    # ax.xaxis.set_major_formatter(myFmt)
                    plt.xlabel('Index')
                    plt.ylabel(col)
                    plt.title(col)
                    plt.legend()
                    # plt.show()
                    plt.savefig(os.path.join(next_directory, col))
                    plt.close()
                # break
        break


def plot_heapmap():
    pass
    # Heatmap
    # sns.heatmap(df_.corr(), annot=True)
    # plt.show()


def plot_pca_data_graph():
    pass
    # PCA: for 1 to 4 conponents
    # plt.figure(figsize=(10, 8))
    # plt.plot(range(1, 4), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    # plt.title('Explained Variance by Components')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.show()


def test_kmeans_n_clusters():
    pass
    # Kmeans + PCA - Test: how many clusters should I need
    wcss = []
    for i in range(1, 21):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_data)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 21), wcss, marker='o', linestyle='--')
    plt.title('K-means with PCA Clustering')
    plt.xlabel('Number of Clustering')
    plt.ylabel('WCSS')
    plt.show()


def test_affinity_propagation_n_clusters():
    pass
    # AffinityPropagation - Test: how many clusters should I need ===== NEED 3.5 TIB ...
    clustering = AffinityPropagation(random_state=42).fit(pca_data)
    print(len(np.unique(clustering.labels_)))


def plot_two_y_axes_by_matplotlib():
    df = pd.read_csv('Jan_processed_data.csv')
    df = df[:1800]
    df = df[df['ingot'] > 350]
    fig, ax = plt.subplots()
    col1 = 'ingot'
    col2 = 'oil_pressure'
    ax.plot(df.index, df[col1], color='red')
    ax.set_xlabel('index', fontsize=14)
    ax.set_ylabel(col1, color='red', fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(df.index, df[col2], color='blue')
    ax2.set_ylabel(col2, color='blue', fontsize=14)
    filename = col1 + '-' + col2
    ax.set_title(filename)
    plt.show()
    # plt.savefig(os.path.join(r'C:\Users\samuello\Downloads\III\旺欉\graph\202203-params plot',
    #                          filename+'(part)'))
    plt.close()


def plot_multiple_y_axes_by_plotly():
    df = pd.read_csv('Jan_processed_data.csv')
    df = df[:1080]

    fig = go.Figure()

    c_list = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#946700']
    y_axes_list = ['ingot', 'discharge', 'mould', 'oil_pressure', 'bucket', 'E_temperature', 'E_humidity']

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ingot'],
        name="ingot",
        marker=dict(size=5, color=c_list[0])
    ))

    for i in range(1, len(y_axes_list)):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[y_axes_list[i]],
            name=y_axes_list[i],
            yaxis="y{}".format(i + 1),
            marker=dict(size=5, color=c_list[i])
        ))

    # Create axis objects
    fig.update_layout(
        xaxis=dict(
            domain=[0.3, 0.7]
        ),
        yaxis=dict(
            title="ingot",
            titlefont=dict(
                color=c_list[0]
            ),
            tickfont=dict(
                color=c_list[0]
            )
        ),
        yaxis2=dict(
            title="discharge",
            titlefont=dict(
                color=c_list[1]
            ),
            tickfont=dict(
                color=c_list[1]
            ),
            anchor="free",
            overlaying="y",
            side="left",
            position=0.15
        ),
        yaxis3=dict(
            title="mould",
            titlefont=dict(
                color=c_list[2]
            ),
            tickfont=dict(
                color=c_list[2]
            ),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        yaxis4=dict(
            title="oil_pressure",
            titlefont=dict(
                color=c_list[3]
            ),
            tickfont=dict(
                color=c_list[3]
            ),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.8
        ),
        yaxis5=dict(
            title="bucket",
            titlefont=dict(
                color=c_list[4]
            ),
            tickfont=dict(
                color=c_list[4]
            ),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.9
        )
    )

    # Update layout properties
    fig.update_layout(
        title_text="multiple y-axes example",
        width=1200,
    )

    fig.show()
    # fig.write_html('plot.html')
