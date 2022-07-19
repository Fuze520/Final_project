from process import *
from sim_matrix import *
from model import *

#%%
# data process
gc.disable()
total_start = time.time()
start = time.time()
all_drugs = read_dataset(dataset_file)
# extract all small molecule drugs from dataset
small_molecules = extract_all_small_molecules(all_drugs)  # the set of all small molecule drugs
# extract all FDA-approved small molecule drugs from small molecule drugs set
small_molecules = extract_all_fda_approved_drugs(small_molecules)
# extract all approved small molecule drugs
small_molecules = extract_all_approved_drugs(small_molecules)
# extract all drugs which have drug-drug interactions
small_molecules = extract_all_drugs_with_ddi(small_molecules)
# extract all drugs which have SMILES
small_molecules = extract_all_drugs_with_SMILES(small_molecules)
small_molecules.pop(7)
small_molecules.pop(73)
count_features(small_molecules)
# save all small molecule drugs info
save_drug_info(small_molecules)
end = time.time()
print(f"All small molecule drugs have been processed.\nTime consuming: {end - start}s.")


# calculate feature similarity
start = time.time()
drugs_file = open("small molecules.pkl", "rb")
all_drugs = pickle.load(drugs_file)

id_file = open("drugs_id.txt", 'w')
name_file = open("drugs_name.txt", 'w')
SMILES_file = open("SMILES.txt", 'w')
target_file = open("target.txt", 'w')
enzyme_file = open("enzyme.txt", 'w')
transporter_file = open("transporter.txt", 'w')
carrier_file = open("carrier.txt", 'w')
pathway_file = open("pathway.txt", 'w')
category_file = open("category.txt", 'w')

targets = list()
enzymes = list()
transporters = list()
carriers = list()
pathways = list()
categories = list()

print(f"{len(all_drugs)} drugs' info have been read.")
for i in tqdm.trange(len(all_drugs), desc="Saving Procession"):
    drug = all_drugs[i]
    id_file.write(drug.id+"\n")
    name_file.write(drug.name+"\n")
    SMILES_file.write(drug.SMILES+"\n")
    targets = list(set(targets).union(set(drug.targets.split("/"))))
    enzymes = list(set(enzymes).union(set(drug.enzymes.split("/"))))
    transporters = list(set(transporters).union(set(drug.transporters.split("/"))))
    carriers = list(set(carriers).union(set(drug.carriers.split("/"))))
    pathways = list(set(pathways).union(set(drug.pathways.split("/"))))
    categories = list(set(categories).union(set(drug.categories.split("/"))))

targets = targets[1:]
enzymes = enzymes[1:]
transporters = transporters[1:]
carriers = carriers[1:]
pathways = pathways[1:]
categories = categories[1:]

print(f"There are {len(targets)} targets, {len(enzymes)} enzymes, {len(transporters)} transporters, "
      f"{len(carriers)} carriers, {len(pathways)} pathways, {len(categories)} categories.")
elems_save2file([targets, enzymes, transporters, carriers, pathways, categories],
                [target_file, enzyme_file, transporter_file, carrier_file, pathway_file, category_file])

id_file.close()
name_file.close()
SMILES_file.close()
target_file.close()
enzyme_file.close()
transporter_file.close()
carrier_file.close()
pathway_file.close()
category_file.close()
drugs_file.close()

end = time.time()
print(f"Time consuming on saving drug info {end-start}s.")

start = time.time()
drugs_file = open("small molecules.pkl", "rb")
all_drugs = pickle.load(drugs_file)

drugs = read_elem_file("drugs_id.txt")
drugs_num = len(drugs)
drug2target = dict()
drug2enzyme = dict()
drug2transporter = dict()
drug2carrier = dict()
drug2category = dict()
drug2pathway = dict()
drug2SMILES = dict()

for i in tqdm.trange(len(all_drugs), desc="Matrix"):
    drug = all_drugs[i]
    drug2target[drug.id] = set(drug.targets.split("/"))
    drug2enzyme[drug.id] = set(drug.enzymes.split("/"))
    drug2transporter[drug.id] = set(drug.transporters.split("/"))
    drug2carrier[drug.id] = set(drug.carriers.split("/"))
    drug2category[drug.id] = set(drug.categories.split("/"))
    drug2pathway[drug.id] = set(drug.pathways.split("/"))
    drug2SMILES[drug.id] = drug.SMILES

sim_target = feature_similarity(drug2target, drugs)
sim_enzyme = feature_similarity(drug2enzyme, drugs)
sim_transporter = feature_similarity(drug2transporter, drugs)
sim_carrier = feature_similarity(drug2carrier, drugs)
sim_pathway = feature_similarity(drug2pathway, drugs)
sim_category = feature_similarity(drug2category, drugs)

mol_set = [Chem.MolFromSmiles(s) for s in list(drug2SMILES.values())]
fp_set = [FingerprintMols.FingerprintMol(mol) for mol in mol_set]
sim_SMILES = np.ones(shape=(drugs_num, drugs_num))

for i in range(drugs_num):
    for j in range(i+1, drugs_num):
        sim_SMILES[i][j] = sim_SMILES[j][i] = DataStructs.TanimotoSimilarity(fp_set[i], fp_set[j])

df_target = pd.DataFrame(sim_target, index=drugs, columns=drugs)
df_target.to_csv("sim_target.csv", index=True)
df_enzyme = pd.DataFrame(sim_enzyme, index=drugs, columns=drugs)
df_enzyme.to_csv("sim_enzyme.csv", index=True)
df_transporter = pd.DataFrame(sim_transporter, index=drugs, columns=drugs)
df_transporter.to_csv("sim_transporter.csv", index=True)
df_carrier = pd.DataFrame(sim_carrier, index=drugs, columns=drugs)
df_carrier.to_csv("sim_carrier.csv", index=True)
df_category = pd.DataFrame(sim_category, index=drugs, columns=drugs)
df_category.to_csv("sim_category.csv", index=True)
df_pathway = pd.DataFrame(sim_pathway, index=drugs, columns=drugs)
df_pathway.to_csv("sim_pathway.csv", index=True)
df_SMILES = pd.DataFrame(sim_SMILES, index=drugs, columns=drugs)
df_SMILES.to_csv("sim_SMILES.csv", index=True)

drugs_file.close()
end = time.time()
print(f"All similarity matrices have been calculated.\nTime consuming: {end-start}s.")

# construct drug-drug interaction network
start = time.time()
drugs = read_elem_file("drugs_id.txt")
drugs_num = len(drugs)
event_set = list()
ddi_matrix = np.zeros(shape=(drugs_num, drugs_num))
df_ddi = pd.DataFrame(ddi_matrix, index=drugs, columns=drugs)
for i in tqdm.trange(len(small_molecules), desc="DDI extraction"):
    drug = small_molecules[i]
    drug1_id = [index.text for index in drug.findall(f"{ns}drugbank-id") if "primary" in index.keys() and bool(index.get("primary"))][0]
    drug1_name = drug.find(f"{ns}name").text.strip()
    for interaction in drug.findall(f"{ns}drug-interactions/{ns}drug-interaction"):
        drug2_id = interaction.findtext(f"{ns}drugbank-id")
        if drug2_id in drugs:
            drug2_name = interaction.findtext(f"{ns}name")
            event = interaction.findtext(f"{ns}description")
            event_set.append((drug1_id, drug1_name, drug2_id, drug2_name, event))
            df_ddi[drug1_id][drug2_id] = 1

event_set = np.array(event_set)
df_event = pd.DataFrame(event_set, columns=['drug1_id', 'drug1_name', 'drug2_id', 'drug2_name', 'event'])
df_event.to_csv("ddi_info.csv", index=False)
df_ddi.to_csv("ddi.csv", index=True)
end = time.time()
print(f"All drug-drug interactions have been extracted.\nTime consuming: {end-start}s.")
del drugs, drugs_num, event_set, ddi_matrix, df_ddi, df_event
del drug, drug1_id, drug1_name, drug2_id, drug2_name, event, interaction
total_end = time.time()
print(f"\nTotal time consuming: {total_end-total_start}s.")
del start, end, total_start, total_end

#%%
df_ddi = pd.read_csv("ddi.csv", index_col=0)
event_set = list()
non_event_set = list()
drugs = df_ddi.columns
df_ddi = df_ddi.values
for i in tqdm.trange(len(drugs), desc="DDI separation"):
    for j in range(i + 1, len(drugs)):
        if df_ddi[i][j] == 0:
            non_event_set.append([i, j])
        else:
            event_set.append([i, j])
print(
    f"There are {len(event_set)} existed drug-drug interaction and {len(non_event_set)} no-existed drug-drug interaction.")

# similarity matrix
sim_SMILES = pd.read_csv("sim_SMILES.csv", index_col=0).values
sim_target = pd.read_csv("sim_target.csv", index_col=0).values
sim_enzyme = pd.read_csv("sim_enzyme.csv", index_col=0).values
sim_transporter = pd.read_csv("sim_transporter.csv", index_col=0).values
sim_category = pd.read_csv("sim_category.csv", index_col=0).values
sim_pathway = pd.read_csv("sim_pathway.csv", index_col=0).values
sim_carrier = pd.read_csv("sim_carrier.csv", index_col=0).values

pca = PCA(n_components=128)
snf_sim = compute.snf([sim_SMILES, sim_target, sim_enzyme, sim_transporter, sim_carrier, sim_pathway, sim_category])
# snf_sim = sim_category
pca.fit(snf_sim)
pca_sim = pca.transform(snf_sim)
features = "All"

seed = 42
random.seed(seed)
event_set = np.array(event_set)
non_event_set = np.array(non_event_set)

event_indices = np.arange(0, len(event_set))
non_event_indices = np.arange(0, len(non_event_set))
random.shuffle(event_indices)
random.shuffle(non_event_indices)

cv_num = 5
model_name = "nn"
if model_name == "nn":
    model_NN(df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num)
else:
    model_other(model_name, df_ddi, event_set, event_indices, non_event_set, non_event_indices, pca_sim, features, cv_num)

#%%
# generate top 10 ddi predictions
df_ddi = pd.read_csv("ddi.csv", index_col=0)
event_set = list()
non_event_set = list()
drugs = df_ddi.columns
df_ddi = df_ddi.values
for i in tqdm.trange(len(drugs), desc="DDI separation"):
    for j in range(i+1, len(drugs)):
        if df_ddi[i][j] == 0:
            non_event_set.append([i, j])
        else:
            event_set.append([i, j])
print(f"There are {len(event_set)} existed drug-drug interaction and {len(non_event_set)} no-existed drug-drug interaction.")


sim_category = pd.read_csv("sim_category.csv", index_col=0).values
pca = PCA(n_components=128)
pca.fit(sim_category)
pca_sim = pca.transform(sim_category)

seed = 42
random.seed(seed)
event_set = np.array(event_set)
non_event_set = np.array(non_event_set)

event_indices = np.arange(0, len(event_set))
non_event_indices = np.arange(0, len(non_event_set))
random.shuffle(event_indices)
random.shuffle(non_event_indices)

cv_num = 5
fold_num1 = len(event_set) // cv_num
fold_num2 = len(non_event_set) // cv_num

y_preds = []
for cv in range(0, cv_num):
    print("\n--> Current cross validation fold is %d <--\n" % (cv+1))
    test_event_indices = event_indices[(cv*fold_num1):((cv+1)*fold_num1)]
    train_event_indices = np.setdiff1d(event_indices, test_event_indices)

    test_non_event_indices = non_event_indices[(cv*fold_num2):((cv+1)*fold_num2)]
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

    X_pred = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            X_pred.append((np.hstack((pca_sim[i], pca_sim[j]))))
    X_pred = np.array(X_pred)
    y_pred = []

    # Multi-layer perception neural network
    model = models.Sequential()
    model.add(layers.Dense(512, activation="relu", input_shape=(128*2, )))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

    history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2)
    print("Model has been trained (fitted).")

    y_pred.append(model.predict(X_pred, batch_size=1000))
    y_preds.append(y_pred)
y_preds = np.array(y_preds)
y_preds = y_preds.reshape((5, 440391, 2))
y_prediction = np.average(y_preds, axis=0)
y_pred_ddi = y_prediction[:, 1]
y_pred_ddi[y_pred_ddi==1] = 0
top10=sorted(range(len(y_pred_ddi)), key=lambda k: y_pred_ddi[k], reverse=True)[:10]
drug_pairs = []
for i in range(len(drugs)):
    for j in range(i + 1, len(drugs)):
        drug_pairs.append([i, j])
top10_pairs = []
for i in top10:
    top10_pairs.append(drug_pairs[i])
