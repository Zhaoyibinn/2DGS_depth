import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d

def draw_depth(theresold_per,l1,method,save_path,data_name):
    theresold_per_np = theresold_per.values
    l1_np = l1.values

    fig, ax = plt.subplots(figsize=(8, 6))
    # fig, ax = plt.subplots()
    ax2 = ax.twinx()
    markers = ['o', 's', '^', 'v', 'p', '*', 'D', 'H']
    colors = ['red', 'blue', 'green', 'cyan', 'purple','orange', 'black', 'magenta', 'lime', 'teal']
    ax.set_ylabel('Depth L1 Error', fontsize=12, color='black')

    ax2.set_ylabel('Correct Percentage', fontsize=12, color='black')
    ax.set_xlabel('Thresholds', fontsize=12, color='black')
    for i in range(theresold_per_np.shape[0]):  
        if method[i+1] in ["2dgs+depth","2dgs+pose"]:
            continue
        
        ax.axhline(y=l1_np[i],color = colors[i],lw=1.0,ls='-.')
        
        f_cubic = interp1d(theresold_np, theresold_per_np[i, :], kind='cubic')
        x_new = np.linspace(theresold_np.min(), theresold_np.max(), 300) 
        y_new = f_cubic(x_new)
        ax2.plot(x_new,y_new , label=f'{method[i+1]}',color = colors[i],zorder=0)
        ax2.scatter(theresold_np, theresold_per_np[i, :],marker=markers[i],color = colors[i])

        

    ax.set_yscale('log') 
        
    ax2.legend(loc='upper left',ncol=6,framealpha=1.0)
    

    # ax.set_title(data_name, fontsize=14, loc='center')
    plt.tight_layout()

    plt.savefig(save_path,dpi=300)
    print(f"{save_path} saved")
    fig.clf()
    return 0


# excel_path = "~/windows_tongbu/文章/新思路/depth.xlsx"
# df = pd.read_excel(excel_path,sheet_name="tum",engine = "openpyxl")
# method = df.iloc[:,0]
# theresold_np = np.array([5,10,20,30,50])

# for i in range(4):
#     data = df.iloc[0:9,6*i+1:6*i+7]

#     data_name = str(data.columns[0])
#     l1 = data.iloc[1:,0]
#     theresold_per = data.iloc[1:,1:6]


#     draw_depth(theresold_per,l1,method,f"vis_lunwen/depth/{data_name}.png","TUM "+data_name)


excel_path = "~/windows_tongbu/文章/新思路/depth.xlsx"
df = pd.read_excel(excel_path,sheet_name="replica",engine = "openpyxl")
method = df.iloc[:,0]
theresold_np = np.array([5,10,20,30,50])

for i in range(3):
    data = df.iloc[0:9,6*i+1:6*i+7]

    data_name = str(data.columns[0])
    l1 = data.iloc[1:,0]
    theresold_per = data.iloc[1:,1:6]


    draw_depth(theresold_per,l1,method,f"vis_lunwen/depth/{data_name}.png","Replica "+data_name)





