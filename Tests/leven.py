import enchant
import jaro
import pandas as pd
import textdistance

data = pd.read_excel (r'../Parameters.ods') 
df = pd.DataFrame(data, columns= ['Product'])
# print (data)


# determining the values of the parameters
string1 = "Nitesh Shah"
string2 = "Shah Nitesh"

print(len(string2))

# the Levenshtein distance between
# string1 and string2
print(1-textdistance.jaccard.normalized_distance(string1, string2))
# print(enchant.utils.levenshtein(string1, string2))

# print(jaro.jaro_winkler_metric(string1, string2))
