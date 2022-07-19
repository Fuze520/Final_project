from utils import *


def read_dataset(file):
    dataset = ET.parse(file)  # read dataset xml file
    drugs = dataset.getroot()  # get all drugs
    print(f"There are {len(drugs)} in total.")
    return drugs


def extract_all_small_molecules(drugs):
    # count the number of biotech drugs and small molecule drugs separately
    # extract all small molecule drugs from dataset
    count_biotech = 0
    count_small_molecule = 0
    extractions = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules"):
        if drugs[i].get("type") == "biotech":
            # count the amount of biotech drugs
            count_biotech += 1
        else:
            # count the amount of small molecule drugs
            extractions.append(drugs[i])
            count_small_molecule += 1
    print(f"There are {count_biotech} biotech drugs.")
    print(f"There are {count_small_molecule} small molecule drugs.")
    return extractions


# extract all FDA-approved small molecule drugs from small molecule drugs set
def extract_all_fda_approved_drugs(drugs):
    extractions = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules(fda)"):
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drugs[i])]))
        if "fda-label" in drug_tags and drugs[i].find(f"{ns}fda-label").text:
            # if the drug has fda-label, this drug is FDA-approved
            extractions.append(drugs[i])
    print(f"There are {len(extractions)} drugs with fda-label.")
    return extractions


# extract all approved small molecule drugs
def extract_all_approved_drugs(drugs):
    extractions = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules(groups)"):
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drugs[i])]))
        if "groups" in drug_tags and "approved" in [elem.text for elem in drugs[i].findall(f"{ns}groups/{ns}group")]:
            extractions.append(drugs[i])
    print(f"There are {len(extractions)} approved drugs.")
    return extractions


# extract all drugs which have drug-drug interactions
def extract_all_drugs_with_ddi(drugs):
    extractions = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules(drug-interactions)"):
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drugs[i])]))
        if "drug-interactions" in drug_tags and len(drugs[i].findall(f"{ns}drug-interactions/{ns}drug-interaction")):
            extractions.append(drugs[i])
    print(f"There are {len(extractions)} drugs with at least one drug-interaction.")
    return extractions


# extract all drugs which have SMILES
def extract_all_drugs_with_SMILES(drugs):
    extractions = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules(SMILES)"):
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drugs[i])]))
        if "calculated-properties" in drug_tags:
            properties = drugs[i].find(f'{ns}calculated-properties').findall(f'{ns}property')
            SMILES = [elem.find(f'{ns}value').text for elem in properties if elem.find(f'{ns}kind').text == 'SMILES']
            if len(SMILES) != 0:
                mol = Chem.MolFromSmiles(SMILES[0])
                if mol:
                    extractions.append(drugs[i])
    print(f"There are {len(extractions)} with SMILES-Mol")
    return extractions


# count each amount of drugs features
def count_features(drugs):
    organism_target = list()
    organism_enzyme = list()
    organism_transporter = list()
    organism_carrier = list()
    for i in tqdm.trange(len(drugs), desc="Protein Info Collection"):
        drug = drugs[i]
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drug)]))

        if "targets" in drug_tags and len(drug.find(f"{ns}targets")) > 0:
            for target in drug.findall(f"{ns}targets/{ns}target"):
                organism = target.findtext(f"{ns}organism")
                if organism not in organism_target:
                    organism_target.append(organism)

        if "enzymes" in drug_tags and len(drug.find(f"{ns}enzymes")) > 0:
            for enzyme in drug.findall(f"{ns}enzymes/{ns}enzyme"):
                organism = enzyme.findtext(f"{ns}organism")
                if organism not in organism_enzyme:
                    organism_enzyme.append(organism)

        if "transporters" in drug_tags and len(drug.find(f"{ns}transporters")) > 0:
            for transporter in drug.findall(f"{ns}transporters/{ns}transporter"):
                organism = transporter.findtext(f"{ns}organism")
                if organism not in organism_transporter:
                    organism_transporter.append(organism)

        if "carriers" in drug_tags and len(drug.find(f"{ns}carriers")) > 0:
            for carrier in drug.findall(f"{ns}carriers/{ns}carrier"):
                organism = carrier.findtext(f"{ns}organism")
                if organism not in organism_carrier:
                    organism_carrier.append(organism)

    print(f"There are {len(organism_target)} organisms in target.\n"
          f"There are {len(organism_enzyme)} organisms in enzyme.\n"
          f"There are {len(organism_transporter)} organisms in transporter.\n"
          f"There are {len(organism_carrier)} organisms in carrier.\n")


def save_drug_info(drugs):
    file = open("small molecules.pkl", "wb")
    drugs_set = list()
    for i in tqdm.trange(len(drugs), desc="Small molecules"):
        drug = drugs[i]
        drug_tags = list(set([elem.tag.replace(ns, "") for elem in list(drug)]))

        drug_id = [index.text for index in drug.findall(f"{ns}drugbank-id") if "primary" in index.keys() and bool(index.get("primary"))][0]
        drug_name = drug.find(f"{ns}name").text.strip()

        properties = drug.find(f'{ns}calculated-properties').findall(f'{ns}property')
        drug_SMILES = [elem.find(f'{ns}value').text for elem in properties if elem.find(f'{ns}kind').text == 'SMILES'][0]

        drug_categories = dict()
        for elem in drug.findall(f"{ns}categories/{ns}category"):
            if elem.findtext(f"{ns}mesh-id").strip():
                drug_categories[elem.findtext(f"{ns}category").strip()] = elem.findtext(f"{ns}mesh-id").strip()
        drug_categories = "/".join(drug_categories.values())

        drug_targets = list()
        if "targets" in drug_tags and len(drug.find(f"{ns}targets")) > 0:
            for target in drug.findall(f"{ns}targets/{ns}target"):
                if target.find(f"{ns}polypeptide") is not None:
                    drug_targets.append(target.find(f"{ns}polypeptide").get("id"))
        drug_targets = "/".join(drug_targets)

        drug_enzymes = list()
        if "enzymes" in drug_tags and len(drug.find(f"{ns}enzymes")) > 0:
            for enzyme in drug.findall(f"{ns}enzymes/{ns}enzyme"):
                if enzyme.find(f"{ns}polypeptide") is not None:
                    drug_enzymes.append(enzyme.find(f"{ns}polypeptide").get("id"))
        drug_enzymes = "/".join(drug_enzymes)

        drug_transporters = list()
        if "transporters" in drug_tags and len(drug.find(f"{ns}transporters")) > 0:
            for transporter in drug.findall(f"{ns}transporters/{ns}transporter"):
                if transporter.find(f"{ns}polypeptide") is not None:
                    drug_transporters.append(transporter.find(f"{ns}polypeptide").get("id"))
        drug_transporters = "/".join(drug_transporters)

        drug_carriers = list()
        if "carriers" in drug_tags and len(drug.find(f"{ns}carriers")) > 0:
            for carrier in drug.findall(f"{ns}carriers/{ns}carrier"):
                if carrier.find(f"{ns}polypeptide") is not None:
                    drug_carriers.append(carrier.find(f"{ns}polypeptide").get("id"))
        drug_carriers = "/".join(drug_carriers)

        drug_pathways = list()
        if "pathways" in drug_tags and len(drug.find(f"{ns}pathways")) > 0:
            drug_pathways = [elem.findtext(f"{ns}smpdb-id") for elem in drug.findall(f"{ns}pathways/{ns}pathway")]
        drug_pathways = "/".join(drug_pathways)

        drug_class = SmallMolecules(drug_id, drug_name,
                                    SMILES=drug_SMILES,
                                    categories=drug_categories,
                                    targets=drug_targets,
                                    enzymes=drug_enzymes,
                                    transporters=drug_transporters,
                                    carriers=drug_carriers,
                                    pathways=drug_pathways)
        drugs_set.append(drug_class)
    pickle.dump(drugs_set, file)
    file.close()
    print(f"{len(drugs_set)} drugs' info have been saved.")
