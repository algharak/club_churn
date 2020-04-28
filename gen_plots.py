from args_pg import parse_args
args = parse_args()
import numpy as np
from matplotlib import pyplot


def flatten_list (l):
    return [item for sublist in l for item in sublist]

def gen_plot(dict):
    tr_crv_title,te_crv_title = dict.keys()
    [crv_error] = dict[tr_crv_title].keys()
    [tr_crv_data] = dict[tr_crv_title].values() ; [te_crv_data] = dict[te_crv_title].values()
    x = np.arange(1,len(tr_crv_data)+1)
    fig, ax = pyplot.subplots(figsize=(8, 8))
    ax.plot(x,tr_crv_data, label='Train')
    ax.plot(x,te_crv_data, label='Test')
    ax.legend()
    pyplot.title(crv_error)
    pyplot.show()
    return

'''
    for plot in plots:
        tr_vs_te.append(plot)
        tr_te_error.append([loss for loss in plot[1].keys()])
        tr_te_data.append(plot[1].values())

        print()
    for i,j in results.items():
        for lossname, data in j.items():
            lossname = j.keys()
            tr_data = j.values()
        te_data = j.values
        fig, ax = pyplot.subplots(figsize=(10, 10))
        ylab = lossname
    exit(0)
        #ax.plot(x_axis,tr_data, label='Train')
        #ax.plot(x_axis,te_data, label='Test')



        #plt.style.use('ggplot')
    return


'''