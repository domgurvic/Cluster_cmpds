def make_canon_smiles(smiles):
    canon_smiles=[]
    for mol in smiles:
        try: 
            canon_smiles.append(Chem.CanonSmiles(mol))
        except:
            return print('Invalid SMILE:', mol)
    return canon_smiles
