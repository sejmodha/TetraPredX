import os
import joblib
import pandas as pd
import FeatureExtractor as ft

infasta, out, tax_label, kmer, cpu, chunk, path = ft.set_vars()

if path is None:
    path = os.getcwd()
else:
    path

print(path)

# print(f'Processing input file {infasta}...\nGenerating feature set...\n')
df = ft.get_feature_table(infasta, out, tax_label, kmer, cpu, chunk)
# print(df.head())

X = df.iloc[:, 0:256]
print(f'Running predictions for input sequences.\nUsing exising models available in {path}')
print('This may take some time....\n')
pred_proba_dict = {}
for file in os.listdir(path):
    if file.endswith(".model"):
        print(file)
        loaded_model = joblib.load(path+'/'+file)
        prefix = file.split('.')[0]
        # print(f'Running predictions using model: {file}')
        pred_proba_dict[prefix] = loaded_model.predict_proba(X)[:, 1]

output_df = pd.DataFrame(pred_proba_dict, index=X.index)
prediction_output_file = f'{out}_predictions.csv'
output_df.to_csv(prediction_output_file, index=True, header=True)

print(f'Analyses completed.\n\nPrediction results saved in {prediction_output_file}.\n')
# print(f'Feature table is saved in {out}_feat.csv.gz\n')
