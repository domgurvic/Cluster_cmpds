def similarity_index(a, b, cutoff): # takes fingerprints and compares them returning list of indexes
    index_similar_comps=[]
    for num in range(0,len(a)):
        for num2 in range(0,len(b)):
            if DataStructs.FingerprintSimilarity(a[num], b[num2]) >= cutoff :
                index_similar_comps.append(num2)
    return index_similar_comps
