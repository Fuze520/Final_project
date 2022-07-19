### 代码使用说明文档

---

1. [引用的包、全局变量、自定义的函数和自定义的类](utils.py)
   - ```imports```
   - ```ns``` （全局变量，该变量的值为DrugBank的网址）
   - ```dataset_file``` （全局变量，该变量的值为DrugBank xml形式的数据集的本地地址）
   - ```class SmallMolecules``` （该类用于存储小分子药物的基本信息）
   - ```def performance_helper(test_num, y_real, y_pred)``` （该函数用于计算预测分类结果的评估指标）
2. [数据预处理 - 提取小分子药物的特征](process.py)
   - ```def read_dataset(file)```（该函数用于读取数据库文件）
   - ```def extract_all_small_molecules(drugs)``` （该函数用于提取所有的小分子药物）
   - ```def extract_all_fda_approved_drugs(drugs)```（提取所有FDA批准的小分子药物）
   - ```def extract_all_approved_drugs(drugs)```（提取所有批准的可以用于临床的小分子药物）
   - ```def extract_all_drugs_with_ddi(drugs)```（提取所有有DDI的小分子药物）
   - ```def extract_all_drugs_with_SMILES(drugs)```（提取所有有SMILES的小分子药物）
   - ```def count_features(drugs)```（计算每种特征的数量）
   - ```def save_drug_info(drugs)```（保存每种药物的信息，包括名称、编号和各种特征）
3. [相似性矩阵的计算](sim_matrix.py)
   - ```def elems_save2file(elem_sets, elem_files)```（用于生成药物-特征对应网络）
   - ```def read_elem_file(elem_file)```（用于读取药物-特征网络）
   - ```def feature_similarity(feature_set, drugs)```（用于计算特征的相似性矩阵）
     - ```def Jaccard(matrix)```（Jaccard Index）
4. [预测模型计算](model.py)
   - ```def model_NN(df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num=5)```（神经网络模型）
   - ```def model_other(model_name, df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num=5)```（其他模型：KNN、LR、RFC）
5. [生成Top10预测结果](main.py)