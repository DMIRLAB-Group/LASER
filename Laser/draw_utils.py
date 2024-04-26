
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def draw_line_pdf(num_line, data, index_X, xlabel, ylabel,data_name=None, ylim=(-0.05, 1.09), loc=(0.7,0.7),
                  file_name = None, is_show=False, is_save=False, title=None, ylable_fontsize="13"
                  ,draw_hline=False):
    """
    df = pd.DataFrame(np.array([[0.2       , 0.4       , 0.9       , 0.94      , 0.94      ,
        0.98      , 1.        ],
       [0.33      , 0.39      , 0.82      , 0.92      , 0.92      ,
        0.99      , 1.        ],
       [0.32666667, 0.39333333, 0.78666667, 0.86666667, 0.85333333,
        0.94666667, 0.99333333],
       [0.325     , 0.465     , 0.785     , 0.855     , 0.83      ,
        0.965     , 0.995     ]]),index=['0.5', '1.0', '1.5', '2.0'],columns=['PCMCI', 'NHPC', 'MLE_SGL', 'ADM4', 'TTHP_NT', 'TTHP_S', 'TTHP'])

    :param num_line:
    :param data:
    :param index_X:
    :param data_name:
    :param xlabel:
    :param ylabel:
    :param file_name:
    :return:
    """
    error_config = {'ecolor': '0.3', 'capsize': 2}
    marker = ['-x', '-D', '-*','-v', '-o',  '-s', '-^']
    color = ['#1f77b4', '#ff7f0e', '#2ca82c', '#9467bd','#8c564b',"#d62728","#e377c2","#7f7f7f"]
    # marker = ['-o','-*']
    marker = marker[:num_line]
    if data_name is None:
        df = pd.DataFrame(data, index=index_X)
    else:
        df = pd.DataFrame(data, index=index_X, columns=data_name,)
    ax = df.plot(kind='line', style=marker,figsize=(5,3), ylim=ylim, rot=0, color=color) # 这里面的line可以改成bar之类的
    from matplotlib.pyplot import MultipleLocator
    y_major_locator = MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid( linestyle="dotted")  # 设置背景线\
    if data_name is None:
        ax.legend().remove()
    if data_name is not None:
        ax.legend(fontsize=5.5, loc=loc)   # 设置图例位置
    ax.set_xlabel(xlabel, fontsize='13')
    ax.set_ylabel(ylabel, fontsize=ylable_fontsize)
    if title is not None:
        plt.title(label=title, fontsize='17', loc='center')
    if draw_hline is True:
        plt.hlines(0,index_X[0],index_X[-1], colors="red")
    plt.xticks(index_X, fontsize=8)
    plt.axhline(y=0, c='black', ls=':')

    if is_save and file_name is not None:
        plt.savefig("graph/" + file_name, format='pdf', bbox_inches='tight')
        df.to_excel("graph/" + file_name + '.xlsx')
    if is_show:
        plt.show()

def readcsv(files):
    data = pd.read_csv(files)
    data_np = np.array(data.values)
    return data_np

def savecsv(data,file_name):
    df = pd.DataFrame(data)
    df.to_excel("data/" + file_name + '.xlsx', encoding="UTF-8")


def draw_bar_pdf(num_line, data, data_std, index_X, data_name, xlabel, ylabel, ylim=(-0.05, 1.09), loc=(0.7,0.7),
                  file_name = None, is_show=False, is_save=False, title=None, bar=True):
    """
    df = pd.DataFrame(np.array([[0.2       , 0.4       , 0.9       , 0.94      , 0.94      ,
        0.98      , 1.        ],
       [0.33      , 0.39      , 0.82      , 0.92      , 0.92      ,
        0.99      , 1.        ],
       [0.32666667, 0.39333333, 0.78666667, 0.86666667, 0.85333333,
        0.94666667, 0.99333333],
       [0.325     , 0.465     , 0.785     , 0.855     , 0.83      ,
        0.965     , 0.995     ]]),index=['0.5', '1.0', '1.5', '2.0'],columns=['PCMCI', 'NHPC', 'MLE_SGL', 'ADM4', 'TTHP_NT', 'TTHP_S', 'TTHP'])

    :param num_line:
    :param data:
    :param index_X:
    :param data_name:
    :param xlabel:
    :param ylabel:
    :param file_name:
    :return:
    """
    error_config = {'ecolor': '0.3', 'capsize': 2}
    marker = ['-x', '-D', '-*', '-s', '-v', '-o', '-^']
    marker = marker[:num_line]
    df = pd.DataFrame(data, index=index_X, columns=data_name)
    df_std = pd.DataFrame(data_std,  index=index_X, columns=data_name)
    if bar == True:
        ax = df.plot(kind='bar', style=marker, yerr=df_std, figsize=(4,2.3), ylim=ylim, rot=0, error_kw=error_config)
    else:

        ax = df.plot(kind='line', style=marker, yerr=df_std, capsline='dash' ,capthick=4, capsize=2,figsize=(4,2.3), ylim=ylim, rot=0)

    ax.grid( linestyle="dotted") # 设置背景线
    ax.legend(fontsize=5,loc='best') # 设置图例位置
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        plt.title(label=title, fontsize='17', loc='center')
    if is_save and file_name is not None:
        plt.savefig("graph/" + file_name + '.pdf', format='pdf', bbox_inches='tight')
        df.to_excel("graph/" + file_name + '.xlsx')
        df_std.to_excel("graph/" + file_name + '_std.xlsx')
    if is_show:
        plt.show()



