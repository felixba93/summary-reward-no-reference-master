import pandas
import matplotlib.pyplot as plt

plt.close('all')


def clean_name(name):
    name = name.replace("'", "")
    name = name.replace(" ", "")
    name = name.replace("\\n", "")
    return name


if __name__ == '__main__':
    # ===set the csv file name
    input_csv = 'outputs/majority_preferences_intra-topic_w-ties_lrate0.1-1e-0.6_seed1-10_balanced_epoch500/majority_intra-topic_w-ties_epoch_500_seed1_balanced.csv'
    # input_csv = 'outputs/all_preferences_intra-topic_w-ties/all_preferences_intra-topic_w-ties.csv'

    # ===here you can include or exclude some cols beforehand
    col_names_of_measures = ['loss_train', 'loss_dev', 'loss_test', 'rho_train',
                             'rho_p_train', 'pcc_train', 'pcc_p_train', 'tau_train', 'tau_p_train',
                             'rho_train_global', 'pcc_train_global', 'tau_train_global',
                             'rho_dev', 'rho_p_dev', 'pcc_dev', 'pcc_p_dev', 'tau_dev', 'tau_p_dev',
                             'rho_dev_global', 'pcc_dev_global', 'tau_dev_global',
                             'rho_test', 'rho_p_test', 'pcc_test', 'pcc_p_test', 'tau_test', 'tau_p_test',
                             'rho_test_global', 'pcc_test_global', 'tau_test_global']

    # ====categories for graphs. separate values according to these columns
    # cat_cols=[] #do not differentiate, plot graphs only for different losses types (variable cols)
    # cat_cols = ['preferences', 'model_type']  # use the model_type, i.e. differentiate between deep and linear
    cat_cols = ['model_type']  # use the model_type, i.e. differentiate between deep and linear
    # ===include or exclude like you want

    cols = []
    # cols+=[col for col in col_names_of_measures] #take all
    cols += [col for col in col_names_of_measures if 'loss' in col]  # take only the losses
    # cols+=[col for col in col_names_of_measures if 'rho' in col] #add rho
    # cols+=[col for col in col_names_of_measures if 'pcc' in col] #add pcc
    # cols += [col for col in col_names_of_measures if 'tau' in col]  # add kendall tau
    # cols+=[col for col in col_names_of_measures if 'global' in col] #add global correlation measures
    cols = [col for col in cols if 'global' not in col]  # exclude the global measures
    cols = [col for col in cols if '_p' not in col]  # exclude the p-values

    # === this makes sense for averaging over several seeds, for instance
    # group_over_episode = True
    group_over_episode = False
    # ===do a scatter plot (preferable if more than one series per curve, i.e., ungrouped data) or line plot
    scatterPlot = True
    # scatterPlot = False

    # === change the color/marker of the graphs for each graph ('always'), for each cat ('cat'), or each loss type ('loss')
    color_change = 'loss'
    marker_change = 'cat'

    cols = sorted(list(set(cols)))
    data = pandas.read_csv(input_csv)
    epochs_col = 'epoch_num'

    # ===clean out column names
    cleaned_names = {name: clean_name(name) for name in data.columns}
    data.rename(columns=cleaned_names, inplace=True)
    print("the cleaned names")
    print(data.columns)

    # ===query/constraints to select the rows for the plot (in the end, there should be rows==no epochs)
    # data = data[data["preferences"] == 'all']  # use this if you only want to plot one of the preference methods
    data = data[data["seed"] == 1]  # use this if you only want to plot one of the seeds
    data = data[data["learn_rate"] == 0.0001]  # use this if you only want to plot one of the seeds
    # data = data[data['model_type'] == 'linear']
    data = data[data['epoch_num'] != 0]  # remove the 0th epoch, which is the one which is random

    # ===do the grouping
    if group_over_episode:
        data = data.groupby('epoch_num', as_index=False).mean()
        # data.reset_index()

    print("number of remaining rows (should be equal to number of epochs): %s" % len(data))
    plt.figure()
    # cols=[data.columns[idx] for idx in show_col]
    #    print(len(data['epoch_num']))
    #    print(len(data['loss_test']))

    # workaround if pandas does not recognize numeric values. just tries to convert everything to numeric
    data = data.apply(pandas.to_numeric, errors='ignore')
    # sometimes the rows are not sorted according to the epochs
    data = data.sort_values(epochs_col)

    ##all possible markers (if you really need to plot a lot)
    # from matplotlib.lines import Line2D
    # markers = [m for m, func in Line2D.markers.items() if func != 'nothing' ]
    # some markers which you iterate over, see https://matplotlib.org/3.2.1/api/markers_api.html#module-matplotlib.markers
    markers = ['o', 's', '*', '+', 'x', 'D', 'v', '<', '>', '^', '.']

    if scatterPlot:
        ax = None
        colorcycle = []
        for catidx, cat in enumerate(data[cat_cols].drop_duplicates().values if len(cat_cols) > 0 else ['']):
            cat_name = " ".join([str(temp) for temp in cat])

            if color_change == 'cat':
                if len(colorcycle) == 0:
                    colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                color = colorcycle.pop()
            if color_change == 'loss':
                colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

            for colidx, col in enumerate(cols):
                print("plotting", cat, col)
                data_temp = data
                # bad hack because i do not know right now how to do it in one row
                # select only the datapoints for the cat selected
                for subcat in zip(cat_cols, cat):
                    data_temp = data_temp[data_temp[subcat[0]] == subcat[1]]
                if color_change == 'always' or color_change == 'loss':
                    if len(colorcycle) == 0:  # start with the cycle
                        colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    color = colorcycle.pop()
                marker = markers[catidx] if marker_change == 'cat' else markers[colidx]
                # plt.scatter(data[epochs_col],data[col])
                ax = data_temp.plot(x=epochs_col, y=col, kind='scatter', color=color, label=cat_name + ' ' + col,
                                    marker=marker, s=2,
                                    ax=ax)
    else:
        data.plot(x=epochs_col, y=cols, kind='line', grid=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(input_csv.replace(".csv", ".pdf"), bbox_inches='tight')
