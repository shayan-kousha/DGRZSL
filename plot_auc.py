import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

paths = {
    "CUB_EASY": [
        "./out/CUB_EASY/Model_1_Eu1_Rls0.001_RWz0.0001_my_model_cosine/",
        "./out/CUB_EASY/Model_2_CAN0.0001_Eu1_Rls0.001_RWz0.0001_CIZSL/"
    ],
    "CUB_HARD": [
        "./out/CUB_HARD/Model_1_Eu1_Rls0.001_RWz0.0001_my_model_cosine/",
        "./out/CUB_HARD/Model_2_CAN0.1_Eu1_Rls0.001_RWz0.0001_CIZSL/",
    ],
    "NAB_EASY": [
        "./out/NAB_EASY/Model_1_Eu1_Rls0.001_RWz0.0001_my_model_cosine/",
        "./out/NAB_EASY/Model_2_CAN1.0_Eu1_Rls0.001_RWz0.0001_CIZSL/",
    ],
    "NAB_HARD": [
        "./out/NAB_HARD/Model_1_Eu1_Rls0.001_RWz0.0001_my_model_cosine/",
        "./out/NAB_HARD/Model_2_CAN0.1_Eu1_Rls0.001_RWz0.0001_CIZSL/"
    ]

}

legend = ["Our Model", "CIZSL", "GAZSL"]
titles = ['CUB easy split', 'CUB hard split', 'NAB easy split', 'NAB hard split']
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
axs = [ax1, ax2, ax3, ax4]
for i, key in enumerate(paths.keys()):
    for path in paths[key]:
        plot_data = np.loadtxt(path + "best_plot.txt")

        acc_S_T_list = plot_data[0]
        acc_U_T_list = plot_data[1]

        axs[i].plot(acc_U_T_list, acc_S_T_list)
    
    axs[i].set(xlabel=r'$A_u$' + "\u2192 T", ylabel=r'$A_s$' + "\u2192 T")
    axs[i].set_title(titles[i],  fontsize=10)
    axs[i].legend(legend)

fig.tight_layout()
fig.savefig('./plot/all_plots')
