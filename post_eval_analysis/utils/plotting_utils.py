import matplotlib.pyplot as plt
import numpy as np

def plot_pred_agreement(full_model_list, display_model_list, metric_name, dataset, conf_matrix):
    '''
    Plot a heatmap where x and y axis are models, the blocks are the prediction agreement
    between the two models
    '''
    full_model_list[0] = ' '
    full_model_list[-1] = ' '
    for idx in range(1, len(full_model_list)):
        for model in display_model_list:
            if model in full_model_list[idx]:
                if full_model_list[idx-1] == ' ' and full_model_list[idx-2] == ' ':
                    full_model_list[idx] = model
                else:
                    full_model_list[idx] = ' '

    plt.imshow(conf_matrix, cmap='GnBu')
    # plt.imshow(conf_matrix, cmap='GnBu', vmin=0, vmax=100)
    plt.colorbar()
    plt.title('Model Pairwise ' + metric_name + ' on ' + dataset)
    ax = plt.gca()
    ax.set_xticks([i for i in range(len(full_model_list))])
    ax.set_xticklabels(full_model_list)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.set_yticks([i for i in range(len(full_model_list))])
    ax.set_yticklabels(full_model_list)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # plt.clf()


def bar_plot(data_to_plot, x_axis_labels, legend_labels, ylabel = 'Number of samples', xlabel='threshold', title='Number of samples for different thresholds (F1)'):
    '''
    Output a barplot for corresponding data, but with customized labels and legends
    '''
    colors = ['#E66100', '#40B0A6', '#FFC107', '#004D40', '#D81B60', '#7570B3', '#0C7BDC', '#785EF0', '#6EA5F7']
    x = np.arange(len(x_axis_labels))  # the label locations
    width = 0.6 / len(data_to_plot)   # the width of the bars, assume there are 
    fig, ax = plt.subplots()
    for data_idx, data_entry in enumerate(data_to_plot):
        rects = ax.bar(x + width * (len(data_to_plot) / 2 - data_idx - 0.5), data_entry, width, label=legend_labels[data_idx], color=colors.pop(), alpha=0.7)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x, x_axis_labels)
    ax.legend()
    plt.show()


def plot_perf_wrt_seq(perfs, seq_lens, seq_type, model_list, metric_name, dataset):
    '''
    Plot the result from perf_wrt_seq_len(), which outputs seq_lens and perfs
    '''
    generative_models = {'t5-v1_1-base', 'bart-base', 't5-v1_1-small'}
    plt.clf()
    x = np.arange(len(seq_lens))  # the label locations
    width = 0.6 / len(model_list)   # the width of the bars

    fig, ax = plt.subplots()
    model_colors = {'bert-base-uncased': '#7570B3', 'bert-base': '#7570B3', 'roberta-base': '#0C7BDC',
    'roberta-large': '#0C7BDC', 'bart-base': '#E66100', 'bart-base-tokenizer': '#999999',
    't5-v1_1-base': '#D81B60', 't5-v1_1-base-tokenizer': '#0C7BDC'}
    colors = ['#7570B3', '#0C7BDC', '#E66100',  '#D81B60', '#004D40', '#40B0A6', '#785EF0', '#FFC107']
    for model_idx, model in enumerate(model_list):
        # rects = ax.bar(x + width * (len(model_list) / 2 - model_idx - 0.5), perfs[model], width, label=model, color=colors[model_idx], alpha=0.7)
        # ax.bar_label(rects, padding=3)
        if model in generative_models:
            marker = 'o'
        else:
            marker = '*'
        plt.plot(x, perfs[model], linestyle='-', marker=marker, color=model_colors[model], label=model, alpha=0.7)
    
    labels = []
    for idx in range(len(seq_lens)):
        if idx == 0:
            labels.append('0-' + str(round(seq_lens[idx], 1)))
        else:
            labels.append(str(round(seq_lens[idx-1], 1)) + '-' + str(round(seq_lens[idx], 1)))
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set_ylabel(metric_name + ' Scores')
    ax.set_xlabel(seq_type)
    # metric_name + ' Scores on ' + dataset + ' by models'
    ax.set_title(metric_name + ' on ' + dataset)
    ax.set_xticks(x, labels)
    ax.legend(prop={'size': 11})
    plt.show()


def plot_model_wrt_datasets(data_to_plot, datasets, full_model_list, y_label_name, x_label_name, title_name):
    '''
    data_to_plot: list [number of models, number of datasets]
    '''
    model_colors = {'bert': '#FFC107', 'roberta': '#40B0A6', 'bart': '#785EF0', 't5': '#0C7BDC'}
    # perf_change = em_change
    # perf_change_ratio = em_change_ratio
    # metric_name = 'EM'

    datasets = list(map(lambda x: x.replace('NaturalQuestions', 'NQ'), datasets))
    x = np.arange(len(datasets))  # the label locations
    width = 0.8 / len(full_model_list)   # the width of the bars
    fig, ax = plt.subplots()
    colors = ['#FFC107', '#40B0A6', '#E66100', '#7570B3', '#0C7BDC', '#D81B60', '#004D40', '#785EF0', '#009e73']
    for model_idx, model in enumerate(full_model_list):
        for model_name in model_colors:
            if model_name in model:
                color = model_colors[model_name]
        rects = ax.bar(x + width * (len(full_model_list) / 2 - model_idx - 0.5), data_to_plot[model_idx], width, label=model, color=color, alpha=0.7)

    ax.set_ylabel(y_label_name)
    ax.set_xlabel(x_label_name)
    ax.set_title(title_name)
    ax.set_xticks(x, datasets, fontsize=9)

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f('s', model_colors[model]) for model in model_colors]
    labels = [model for model in model_colors]
    ax.legend(handles, labels)
    plt.show()