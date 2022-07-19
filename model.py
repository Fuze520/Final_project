from utils import *


def model_other(model_name, df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num=5):
    fold_num1 = len(event_set) // cv_num
    fold_num2 = len(non_event_set) // cv_num
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    start = time.time()
    results = list()
    results_columns = ["accuracy", "recall", "precision", "specificity", "f1_score"]
    for cv in range(0, cv_num):
        print("\n--> Current cross validation fold is %d <--\n" % (cv + 1))
        test_event_indices = event_indices[(cv * fold_num1):((cv + 1) * fold_num1)]
        train_event_indices = np.setdiff1d(event_indices, test_event_indices)

        test_non_event_indices = non_event_indices[(cv * fold_num2):((cv + 1) * fold_num2)]
        train_non_event_indices = np.setdiff1d(non_event_indices, test_non_event_indices)

        test_set = np.concatenate((event_set[test_event_indices], non_event_set[test_non_event_indices]), axis=0)
        train_set = np.concatenate((event_set[train_event_indices], non_event_set[train_non_event_indices]), axis=0)
        random.shuffle(test_set)
        random.shuffle(train_set)

        X_train = []
        y_train = []
        for i in tqdm.trange(len(train_set), desc="Train dataset"):
            # X_train.append(np.hstack((np.hstack((df_ddi[train_set[i, 0]], df_ddi[train_set[i, 1]])),
            #                           np.hstack((sim_snf[train_set[i, 0]], sim_snf[train_set[i, 1]])))))
            X_train.append(np.hstack((pca_sim[train_set[i, 0]], pca_sim[train_set[i, 1]])))
            y_train.append(df_ddi[train_set[i, 0]][train_set[i, 1]])
        X_train = np.array(X_train).astype(dtype='float32')
        y_train = np.array(y_train).astype(dtype='float32')
        # y_train = utils.to_categorical(y_train, num_classes=2)
        print("Train dataset has been set.")

        X_test = []
        y_test = []
        for i in tqdm.trange(len(test_set), desc="Test dataset"):
            # X_test.append(np.hstack((np.hstack((df_ddi[test_set[i, 0]], df_ddi[test_set[i, 1]])),
            #                          np.hstack((sim_snf[test_set[i, 0]], sim_snf[test_set[i, 1]])))))
            X_test.append(np.hstack((pca_sim[test_set[i, 0]], pca_sim[test_set[i, 1]])))
            y_test.append(df_ddi[test_set[i, 0]][test_set[i, 1]])
        X_test = np.array(X_test).astype(dtype='float32')
        y_test = np.array(y_test).astype(dtype='float32')
        # y_test = utils.to_categorical(y_test, num_classes=2)
        print("Test dataset has been set.")

        if model_name == 'knn':
            clf = KNeighborsClassifier(2)
        elif model_name == 'lr':
            clf = LogisticRegression(max_iter=len(X_train))
        elif model_name == 'rfc':
            clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        print("Model has been trained (fitted).")

        y_pred = clf.predict(X_test)
        acc, recall, precision, specificity, f1score = performance_helper(len(X_test), y_test, y_pred)
        results.append([acc, recall, precision, specificity, f1score])
        print(f"\naccuracy={acc}\n"
              f"recall={recall}\n"
              f"precision={precision}\n"
              f"specificity={specificity}\n"
              f"f1score={f1score}\n")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold-%d(area=%0.4f)" % (cv + 1, roc_auc))

    df_results = pd.DataFrame(results, columns=results_columns)
    df_results.to_csv(f"./results/ROC({features}).csv")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.4f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC ({features})')
    plt.legend(loc='lower right')
    plt.savefig(f"./results/ROC({features}).png")
    plt.show()

    end = time.time()
    print()
    print(np.average(results, axis=0))
    file = open("./results/result.txt", "a")
    file.write(features)
    file.write("\n")
    file.write(str(mean_auc))
    for i in np.average(results, axis=0):
        file.write(" ")
        file.write(str(i))
    file.write(" ")
    file.write(str(end - start) + "s\n")
    # file.write("\n")
    file.close()


def model_NN(df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num=5):
    fold_num1 = len(event_set) // cv_num
    fold_num2 = len(non_event_set) // cv_num

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # binary_acc = list()
    # val_binary_acc = list()
    # loss = list()
    # val_loss = list()
    start = time.time()
    results = list()
    results_columns = ["accuracy", "recall", "precision", "specificity", "f1_score"]
    for cv in range(0, cv_num):
        print("\n--> Current cross validation fold is %d <--\n" % (cv+1))
        test_event_indices = event_indices[(cv*fold_num1):((cv+1)*fold_num1)]
        train_event_indices = np.setdiff1d(event_indices, test_event_indices)

        test_non_event_indices = non_event_indices[(cv*fold_num1):((cv+1)*fold_num2)]
        train_non_event_indices = np.setdiff1d(non_event_indices, test_non_event_indices)

        test_set = np.concatenate((event_set[test_event_indices], non_event_set[test_non_event_indices]), axis=0)
        train_set = np.concatenate((event_set[train_event_indices], non_event_set[train_non_event_indices]), axis=0)
        random.shuffle(test_set)
        random.shuffle(train_set)

        X_train = []
        y_train = []
        for i in tqdm.trange(len(train_set), desc="Train dataset"):
            # X_train.append(np.hstack((np.hstack((df_ddi[train_set[i, 0]], df_ddi[train_set[i, 1]])),
            #                           np.hstack((sim_snf[train_set[i, 0]], sim_snf[train_set[i, 1]])))))
            X_train.append(np.hstack((pca_sim[train_set[i, 0]], pca_sim[train_set[i, 1]])))
            y_train.append(df_ddi[train_set[i, 0]][train_set[i, 1]])
        X_train = np.array(X_train).astype(dtype='float32')
        y_train = np.array(y_train).astype(dtype='float32')
        y_train = utils.to_categorical(y_train, num_classes=2)
        print("Train dataset has been set.")

        X_test = []
        y_test = []
        for i in tqdm.trange(len(test_set), desc="Test dataset"):
            # X_test.append(np.hstack((np.hstack((df_ddi[test_set[i, 0]], df_ddi[test_set[i, 1]])),
            #                          np.hstack((sim_snf[test_set[i, 0]], sim_snf[test_set[i, 1]])))))
            X_test.append(np.hstack((pca_sim[test_set[i, 0]], pca_sim[test_set[i, 1]])))
            y_test.append(df_ddi[test_set[i, 0]][test_set[i, 1]])
        X_test = np.array(X_test).astype(dtype='float32')
        y_test = np.array(y_test).astype(dtype='float32')
        y_test = utils.to_categorical(y_test, num_classes=2)
        print("Test dataset has been set.")

        # Multi-layer perception neural network
        model = models.Sequential()
        # model.add(layers.Dense(256, activation="relu", input_shape=(128*2, )))
        model.add(layers.Dense(512, activation="relu", input_shape=(128*2, )))
        # model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        # model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

        # early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="auto")
        history = model.fit(X_train, y_train, batch_size=128, epochs=200, validation_split=0.2)
        print("Model has been trained (fitted).")
        # model.save(f"./models/5layersDNN(Fold{cv}).h5")

        # binary_acc.append(history.history['binary_accuracy'])
        # val_binary_acc.append(history.history['val_binary_accuracy'])
        # loss.append(history.history['loss'])
        # val_loss.append(history.history['val_loss'])

        y_pred = model.predict(X_test, batch_size=128)
        y_pred = np.argmax(y_pred, axis=1)

        y_test = np.argmax(y_test, axis=1)
        acc, recall, precision, specificity, f1score = performance_helper(len(X_test), y_test, y_pred)
        results.append([acc, recall, precision, specificity, f1score])
        print(f"\naccuracy={acc}\n"
              f"recall={recall}\n"
              f"precision={precision}\n"
              f"specificity={specificity}\n"
              f"f1score={f1score}\n")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold-%d(area=%0.4f)" % (cv+1, roc_auc))

    df_results = pd.DataFrame(results, columns=results_columns)
    df_results.to_csv(f"./results/ROC({features}).csv")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.4f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
    # plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC ({features})')
    plt.legend(loc='lower right')
    plt.savefig(f"./results/ROC({features}).png")
    plt.show()

    end = time.time()
    print()
    print(np.average(results, axis=0))
    file = open("./results/result.txt", "a")
    file.write(features)
    file.write("\n")
    file.write(str(mean_auc))
    for i in np.average(results, axis=0):
        file.write(" ")
        file.write(str(i))
    file.write(" ")
    file.write(str(end-start)+"s\n")
    # file.write("\n")
    file.close()


    # def plot_annotation_line():
    #     plt.axvline(x=5, c="black", ls="--", lw=0.8)
    #     plt.axvline(x=10, c="black", ls="--", lw=0.8)
    #     plt.axvline(x=15, c="red", ls="--", lw=1)
    #     # plt.axvline(x=15, c="black", ls="--", lw=0.8)
    #     plt.axvline(x=20, c="black", ls="--", lw=0.8)
    #
    #
    # # summarize history for accuracy
    # plt.rcParams['font.family'] = ["SimHei"]
    # plt.plot(np.average(binary_acc, axis=0))
    # plt.plot(np.average(val_binary_acc, axis=0))
    # plt.title('5层神经网络模型的训练准确率')
    # plt.ylabel('二分类准确率')
    # plt.xlabel('Epoch')
    # plt.legend(['训练', '验证'], loc='lower right')
    # plot_annotation_line()
    # plt.axhline(y=np.average(val_binary_acc, axis=0)[15], c="red", ls="--", lw=1)
    # plt.axhline(y=np.average(binary_acc, axis=0)[15], c="red", ls="--", lw=1)
    # # plt.savefig("model acc in 5-layer 200-epoch.png")
    # plt.savefig("5层神经网络模型的训练准确率（200 epochs）.png")
    # plt.show()
    #
    # # summarize history for loss
    # plt.rcParams['font.family'] = ["SimHei"]
    # plt.plot(np.average(loss, axis=0))
    # plt.plot(np.average(val_loss, axis=0))
    # plt.title('5层神经网络模型的训练损失率')
    # plt.ylabel('损失率')
    # plt.xlabel('Epoch')
    # plt.legend(['训练', '验证'], loc='center right')
    # plot_annotation_line()
    # plt.axhline(y=np.average(val_loss, axis=0)[15], c="red", ls="--", lw=1)
    # plt.axhline(y=np.average(loss, axis=0)[15], c="red", ls="--", lw=1)
    # # plt.savefig("model loss in 5-layer 200-epoch.png")
    # plt.savefig("5层神经网络模型的训练损失率（200 epochs）.png")
    # plt.show()
