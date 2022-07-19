from utils import *


def elems_save2file(elem_sets, elem_files):
    for i in range(len(elem_sets)):
        for elem in elem_sets[i]:
            elem_files[i].write(elem+"\n")


def read_elem_file(elem_file):
    f = open(elem_file, 'r')
    elem_set = list()
    for line in f.readlines():
        elem_set.append(line.strip("\n"))
    f.close()
    return elem_set


def jaccard_similarity(set_a, set_b):
    sim = float(len(set_a.intersection(set_b))) / len(set_a.union(set_b))
    return sim


def feature_similarity(feature_set, drugs):
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_features = list()
    for i in range(len(drugs)):
        drug_features = list(feature_set[drugs[i]])
        for feature in drug_features:
            if feature not in all_features:
                all_features.append(feature)
    feature_matrix = np.zeros((len(drugs), len(all_features)), dtype=float)
    df_feature = pd.DataFrame(feature_matrix, columns=all_features)
    for i in range(len(drugs)):
        drug_features = list(feature_set[drugs[i]])
        for feature in drug_features:
            df_feature[feature].iloc[i] = 1
    sim_matrix = Jaccard(df_feature.values)

    return sim_matrix
