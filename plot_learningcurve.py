import pandas
import matplotlib.pyplot as plt
plt.close('all')

def clean_name(name):
    name=name.replace("'","")
    name=name.replace(" ","")
    name=name.replace("\\n","")
    return name

if __name__ == '__main__':
    #===set the csv file name
    input_csv='BetterRewardsStatistics_v4.csv'

    #===here you can include or exclude some cols beforehand
    col_names_of_measures=[ 'loss_train', 'loss_dev', 'loss_test',
       'rho_dev', 'pcc_dev', 'tau_dev', 'rho_dev_global', 'pcc_dev_global',
       'tau_dev_global', 'rho_test', 'pcc_test', 'tau_test', 'rho_test_global',
       'pcc_test_global', 'tau_test_global', 'rho_train', 'pcc_train',
       'tau_train', 'rho_train_global', 'pcc_train_global',
       'tau_train_global']

    #===include or exclude like you want

    cols=[]
    #cols+=[col for col in col_names_of_measures] #take all
    cols+=[col for col in col_names_of_measures if 'loss' in col] #take only the losses
    #cols+=[col for col in col_names_of_measures if 'rho' in col] #add rho
    #cols+=[col for col in col_names_of_measures if 'global' in col] #add global correlation measures
    #cols=[col for col in cols if 'global' not in col] #exclude the global measures

    #=== this makes sense for averaging over several seeds, for instance
    group_over_episode=True
    #group_over_episode=False

    #===do a scatter plot (preferable if more than one series per curve, i.e., ungrouped data) or line plot
    #scatterPlot=True
    scatterPlot=False

    cols=list(set(cols))
    data=pandas.read_csv(input_csv)
    epochs_col='epoch_num'

    #===clean out column names
    cleaned_names={name:clean_name(name) for name in data.columns}
    data.rename(columns= cleaned_names,inplace=True)
    print("the cleaned names")
    print(data.columns)

    #===query/constraints to select the rows for the plot (in the end, there should be rows==no epochs)
    #data=data[data["seed"]==1] #use this if you only want to plot one of the seeds
    data=data[data['model_type']=='linear']
    #data=data[data['epoch_num']==1]

    #===do the grouping
    if group_over_episode:
        data=data.groupby('epoch_num', as_index=False).mean()
        #data.reset_index()

    print("number of remaining rows (should be equal to number of epochs): %s"%len(data))
    plt.figure()
    #cols=[data.columns[idx] for idx in show_col]
#    print(len(data['epoch_num']))
#    print(len(data['loss_test']))
    if scatterPlot:
        ax=None
        colorcycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for col in cols:
            ax=data.plot(x='epoch_num',y=col,kind='scatter',color=colorcycle.pop(),label=col,ax=ax)
    else:
        data.plot(x='epoch_num',y=cols,kind='line',grid=True)
    plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.savefig(input_csv.replace(".csv",".pdf"),bbox_inches='tight')