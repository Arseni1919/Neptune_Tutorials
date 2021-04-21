# This is a sample Python script.
import neptune.new as neptune
import plotly
import plotly.express as px
from neptune.new.types import File
import time
import numpy as np
import pandas as pd
import seaborn as sns


run = neptune.init(project='1919ars/Neptune-Tutorials', tags=['tutorials', 'try'],
                   name='BlaBliBlu')

PARAMS = {'epoch_nr': 100,
          'lr': 0.005,
          'use_nesterov': True}


run['parameters'] = PARAMS

# You can also specify parameters one by one
run['my_params/batch_size'] = 64

run['my_params/batch_size'] = 128

# Update lr value
run['my_params/lr'] = 0.007


# fig.show()
# fig.write_image("fig1.png")

# neptune.log_artifact('model_viz.png')
run['model/viz'].upload("fig1.png")

# neptune.log_artifact('model.pt')
# run['trained_model'].upload('model.pt')

# neptune.download_artifact('model.pt')
# run['trained_model'].download()

# plotly
fig = px.line([1,2,10,100])
run['int_chart'].upload(File.as_html(fig))

# pandas
iris = sns.load_dataset('iris')
run['pred_df'].upload(File.as_html(iris))


a = np.array([1, 4, 20, 400])
for i in list(range(10)):
    x_1 = i**2
    x_2 = np.log(i+1)
    x_3 = np.exp(i+1)
    run['acc'].log(x_1)
    run['acc_log'].log(f'{x_1}')
    run['train/acc'].log(x_2)
    run['train/acc2'].log(x_3)
    run['misclasified1'].log(File('fig1.png'))  # supported only for images
    time.sleep(5)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Finished, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
