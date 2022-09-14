import torch
import pandas as pd
import numpy as np
import seaborn as sns

def evaluation(model, df, sequences, criterion, col_out):
    res = {}
    prediction = []
    timestamp = []
    loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, seq in enumerate(sequences):
            pred_seq = model(seq[0].unsqueeze(0).to(device)).squeeze(0)

            loss.append(criterion(seq[1], pred_seq).item())

            pred_seq = np.array(pred_seq.detach().cpu())
            prediction.append(pred_seq)
            #timestamp.extend(range(i,i+seq.size(1)))
            time_window = seq[0].size(0)
            len_forecast = seq[1].size(0)
            timestamp_seq = list(df['timestamp'][time_window+i:time_window+i+len_forecast])
            timestamp.extend(timestamp_seq)
    res = pd.DataFrame(prediction, columns = col_out)
    res['timestamp'] = timestamp
    
    return res, loss.mean()


def plot(df, res, col_out):
       plot_df = res['timestamp', col_out]
       plot_df = plot_df.rename(columns = {col_out : 'pred'})
       plot_df = plot_df.merge(df['timestamp', col_out], on = 'timestamp', how = 'left')
       df_melted = plot_df.melt(id_vars = ['timestamp'], value_vars = ['true', 'pred'])
       sns.lineplot(data = df_melted, x = 'timestamp', y = 'value', hue = 'variable')