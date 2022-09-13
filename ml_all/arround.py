import glob
import pandas as pd



for l in range(1):
    files = glob.glob('C:\\Users\\rikua\\Documents\\ml_run_fdToWl\\ml_all\\now_all\\*')
    test = [files[1],files[5],files[7],files[9],files[11],files[14],files[17],files[19],files[22],files[24]]
    path ="C:\\Users\\rikua\\Documents\\ml_run_fdToWl\\ml_all\\output\\"
    df_list_test = []
    df_list_train = []
    df2 = []
    for file in files:
        if file in test:
            sub_df = pd.read_csv(file)
            df_list_test.append(sub_df)
        else :
            sub_df = pd.read_csv(file)
            df_list_train.append(sub_df)

    df_list_train = pd.concat(df_list_train)
    df_list_test = pd.concat(df_list_test)
    """
    for i in range(len(df_list_train)-1):
        if df_list_train.iat[i,25] != df_list_train.iat[i+1,25]:
            if s == 0:
                df2 = df_list_train.iloc[i-100:i , :]
            else:
                df2.concat(df_list_train.iloc[i-100:i , :])
            print(df_list_train.iloc[i-100:i , :])
    """
    for i in range(len(df_list_train)//300-1):
        df2.append(df_list_train.iloc[i*300:i*300+100, :])
        #print(1)
        print(df_list_train.iloc[i*300:i*300+100, :])
    df_list_train = pd.concat(df2)

    for i in range(len(df_list_test)//300-1):
        df2.append(df_list_test.iloc[i*300:i*300+100, :])
        #print(1)
        print(df_list_test.iloc[i*300:i*300+100, :])
    df_list_test = pd.concat(df2)



    print(df2)

    train_x = df_list_train.drop("workload",axis=1)
    train_y= df_list_train[["workload"]]
    test_x=df_list_test.drop("workload",axis=1)
    test_y= df_list_test[["workload"]]
    


    train2 = pd.concat([train_x,train_y], axis=1)
    test2 = pd.concat([test_x, test_y], axis=1)
    print(train2)
    print(test2)



