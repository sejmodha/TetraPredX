import TrainModels as tm

# set variables
df, target_column, cpu, out, path, cv  = tm.set_vars()

data_dict = tm.get_train_test(df, target_column, 0.3, 4, 256)

# Train binary classification models
tm.train_models_rfc(data_dict, out, path, cpu, cv)
