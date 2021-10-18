import pandas as pd
import enchant
import jaro
import textdistance

from fastDamerauLevenshtein import damerauLevenshtein

df = pd.read_excel('../Algo_comparison.ods')
k=0
for i, j in zip(df.OG_address_fields, df.Erroneous_fields):
    k+=1
    # print(enchant.utils.levenshtein(str(i), str(j)))
    # print(jaro.jaro_winkler_metric(str(i), str(j)))
    
    # print(damerauLevenshtein(str(i), str(j), similarity=False))  
    print(textdistance.monge_elkan.distance(str(i), str(j)))

print(k)

# print(df.Erroneous_fields)