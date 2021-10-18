from numpy.lib.function_base import average
import textdistance
import numpy as np
import pandas as pd
import math

algo="levenshtein"


def suffixDict():
    """
    Use common abbreviations -> USPS standardized abbreviation to replace common street suffixes

    Obtains list from https://www.usps.com/send/official-abbreviations.htm
    """
    return {'trpk': 'tpke', 'forges': 'frgs', 'bypas': 'byp', 'mnr': 'mnr', 'viaduct': 'via', 'mnt': 'mt',
            'lndng': 'lndg', 'vill': 'vlg', 'aly': 'aly', 'mill': 'ml', 'pts': 'pts', 'centers': 'ctrs', 'row': 'row', 'cnter': 'ctr',
            'hrbor': 'hbr', 'tr': 'trl', 'lndg': 'lndg', 'passage': 'psge', 'walks': 'walk', 'frks': 'frks', 'crest': 'crst', 'meadows': 'mdws',
            'freewy': 'fwy', 'garden': 'gdn', 'bluffs': 'blfs', 'vlg': 'vlg', 'vly': 'vly', 'fall': 'fall', 'trk': 'trak', 'squares': 'sqs',
            'trl': 'trl', 'harbor': 'hbr', 'frry': 'fry', 'div': 'dv', 'straven': 'stra', 'cmp': 'cp', 'grdns': 'gdns', 'villg': 'vlg',
            'meadow': 'mdw', 'trails': 'trl', 'streets': 'sts', 'prairie': 'pr', 'hts': 'hts', 'crescent': 'cres', 'pass': 'pass',
            'ter': 'ter', 'port': 'prt', 'bluf': 'blf', 'avnue': 'ave', 'lights': 'lgts', 'rpds': 'rpds', 'harbors': 'hbrs',
            'mews': 'mews', 'lodg': 'ldg', 'plz': 'plz', 'tracks': 'trak', 'path': 'path', 'pkway': 'pkwy', 'gln': 'gln',
            'bot': 'btm', 'drv': 'dr', 'rdg': 'rdg', 'fwy': 'fwy', 'hbr': 'hbr', 'via': 'via', 'divide': 'dv', 'inlt': 'inlt',
            'fords': 'frds', 'avenu': 'ave', 'vis': 'vis', 'brk': 'brk', 'rivr': 'riv', 'oval': 'oval', 'gateway': 'gtwy',
            'stream': 'strm', 'bayoo': 'byu', 'msn': 'msn', 'knoll': 'knl', 'expressway': 'expy', 'sprng': 'spg',
            'flat': 'flt', 'holw': 'holw', 'grden': 'gdn', 'trail': 'trl', 'jctns': 'jcts', 'rdgs': 'rdgs',
            'tunnel': 'tunl', 'ml': 'ml', 'fls': 'fls', 'flt': 'flt', 'lks': 'lks', 'mt': 'mt', 'groves': 'grvs',
            'vally': 'vly', 'ferry': 'fry', 'parkway': 'pkwy', 'radiel': 'radl', 'strvnue': 'stra', 'fld': 'fld',
            'overpass': 'opas', 'plaza': 'plz', 'estate': 'est', 'mntn': 'mtn', 'lock': 'lck', 'orchrd': 'orch',
            'strvn': 'stra', 'locks': 'lcks', 'bend': 'bnd', 'kys': 'kys', 'junctions': 'jcts', 'mountin': 'mtn',
            'burgs': 'bgs', 'pine': 'pne', 'ldge': 'ldg', 'causway': 'cswy', 'spg': 'spg', 'beach': 'bch', 'ft': 'ft',
            'crse': 'crse', 'motorway': 'mtwy', 'bluff': 'blf', 'court': 'ct', 'grov': 'grv', 'sprngs': 'spgs',
            'ovl': 'oval', 'villag': 'vlg', 'vdct': 'via', 'neck': 'nck', 'orchard': 'orch', 'light': 'lgt',
            'sq': 'sq', 'pkwy': 'pkwy', 'shore': 'shr', 'green': 'grn', 'strm': 'strm', 'islnd': 'is',
            'turnpike': 'tpke', 'stra': 'stra', 'mission': 'msn', 'spngs': 'spgs', 'course': 'crse',
            'trafficway': 'trfy', 'terrace': 'ter', 'hway': 'hwy', 'avenue': 'ave', 'glen': 'gln',
            'boul': 'blvd', 'inlet': 'inlt', 'la': 'ln', 'ln': 'ln', 'frst': 'frst', 'clf': 'clf',
            'cres': 'cres', 'brook': 'brk', 'lk': 'lk', 'byp': 'byp', 'shoar': 'shr', 'bypass': 'byp',
            'mtin': 'mtn', 'ally': 'aly', 'forest': 'frst', 'junction': 'jct', 'views': 'vws', 'wells': 'wls', 'cen': 'ctr',
            'exts': 'exts', 'crt': 'ct', 'corners': 'cors', 'trak': 'trak', 'frway': 'fwy', 'prarie': 'pr', 'crossing': 'xing',
            'extn': 'ext', 'cliffs': 'clfs', 'manors': 'mnrs', 'ports': 'prts', 'gatewy': 'gtwy', 'square': 'sq', 'hls': 'hls',
            'harb': 'hbr', 'loops': 'loop', 'mdw': 'mdw', 'smt': 'smt', 'rd': 'rd', 'hill': 'hl', 'blf': 'blf',
            'highway': 'hwy', 'walk': 'walk', 'clfs': 'clfs', 'brooks': 'brks', 'brnch': 'br', 'aven': 'ave',
            'shores': 'shrs', 'iss': 'iss', 'route': 'rte', 'wls': 'wls', 'place': 'pl', 'sumit': 'smt', 'pines': 'pnes',
            'trks': 'trak', 'shoal': 'shl', 'strt': 'st', 'frwy': 'fwy', 'heights': 'hts', 'ranches': 'rnch',
            'boulevard': 'blvd', 'extnsn': 'ext', 'mdws': 'mdws', 'hollows': 'holw', 'vsta': 'vis', 'plains': 'plns',
            'station': 'sta', 'circl': 'cir', 'mntns': 'mtns', 'prts': 'prts', 'shls': 'shls', 'villages': 'vlgs',
            'park': 'park', 'nck': 'nck', 'rst': 'rst', 'haven': 'hvn', 'turnpk': 'tpke', 'expy': 'expy', 'sta': 'sta',
            'expr': 'expy', 'stn': 'sta', 'expw': 'expy', 'street': 'st', 'str': 'st', 'spurs': 'spur', 'crecent': 'cres',
            'rad': 'radl', 'ranch': 'rnch', 'well': 'wl', 'shoals': 'shls', 'alley': 'aly', 'plza': 'plz', 'medows': 'mdws',
            'allee': 'aly', 'knls': 'knls', 'ests': 'ests', 'st': 'st', 'anx': 'anx', 'havn': 'hvn', 'paths': 'path', 'bypa': 'byp',
            'spgs': 'spgs', 'mills': 'mls', 'parks': 'park', 'byps': 'byp', 'flts': 'flts', 'tunnels': 'tunl', 'club': 'clb', 'sqrs': 'sqs',
            'hllw': 'holw', 'manor': 'mnr', 'centre': 'ctr', 'track': 'trak', 'hgts': 'hts', 'rnch': 'rnch', 'crcle': 'cir', 'falls': 'fls',
            'landing': 'lndg', 'plaines': 'plns', 'viadct': 'via', 'gdns': 'gdns', 'gtwy': 'gtwy', 'grove': 'grv', 'camp': 'cp', 'tpk': 'tpke',
            'drive': 'dr', 'freeway': 'fwy', 'ext': 'ext', 'points': 'pts', 'exp': 'expy', 'ky': 'ky', 'courts': 'cts', 'pky': 'pkwy', 'corner': 'cor',
            'crssing': 'xing', 'mnrs': 'mnrs', 'unions': 'uns', 'cyn': 'cyn', 'lodge': 'ldg', 'trfy': 'trfy', 'circle': 'cir', 'bridge': 'brg',
            'dl': 'dl', 'dm': 'dm', 'express': 'expy', 'tunls': 'tunl', 'dv': 'dv', 'dr': 'dr', 'shr': 'shr', 'knolls': 'knls', 'greens': 'grns',
            'tunel': 'tunl', 'fields': 'flds', 'common': 'cmn', 'orch': 'orch', 'crk': 'crk', 'river': 'riv', 'shl': 'shl', 'view': 'vw',
            'crsent': 'cres', 'rnchs': 'rnch', 'crscnt': 'cres', 'arc': 'arc', 'btm': 'btm', 'blvd': 'blvd', 'ways': 'ways', 'radl': 'radl',
            'rdge': 'rdg', 'causeway': 'cswy', 'parkwy': 'pkwy', 'juncton': 'jct', 'statn': 'sta', 'gardn': 'gdn', 'mntain': 'mtn',
            'crssng': 'xing', 'rapid': 'rpd', 'key': 'ky', 'plns': 'plns', 'wy': 'way', 'cor': 'cor', 'ramp': 'ramp', 'throughway': 'trwy',
            'estates': 'ests', 'ck': 'crk', 'loaf': 'lf', 'hvn': 'hvn', 'wall': 'wall', 'hollow': 'holw', 'canyon': 'cyn', 'clb': 'clb',
            'cswy': 'cswy', 'village': 'vlg', 'cr': 'crk', 'trce': 'trce', 'cp': 'cp', 'cv': 'cv', 'ct': 'cts', 'pr': 'pr', 'frg': 'frg',
            'jction': 'jct', 'pt': 'pt', 'mssn': 'msn', 'frk': 'frk', 'brdge': 'brg', 'cent': 'ctr', 'spur': 'spur', 'frt': 'ft', 'pk': 'park',
            'fry': 'fry', 'pl': 'pl', 'lanes': 'ln', 'gtway': 'gtwy', 'prk': 'park', 'vws': 'vws', 'stravenue': 'stra', 'lgt': 'lgt',
            'hiway': 'hwy', 'ctr': 'ctr', 'prt': 'prt', 'ville': 'vl', 'plain': 'pln', 'mount': 'mt', 'mls': 'mls', 'loop': 'loop',
            'riv': 'riv', 'centr': 'ctr', 'is': 'is', 'prr': 'pr', 'vl': 'vl', 'avn': 'ave', 'vw': 'vw', 'ave': 'ave', 'spng': 'spg',
            'hiwy': 'hwy', 'dam': 'dm', 'isle': 'isle', 'crcl': 'cir', 'sqre': 'sq', 'jct': 'jct', 'jctn': 'jct', 'mountain': 'mtn',
            'keys': 'kys', 'parkways': 'pkwy', 'drives': 'drs', 'tunl': 'tunl', 'jcts': 'jcts', 'knl': 'knl', 'center': 'ctr',
            'driv': 'dr', 'tpke': 'tpke', 'sumitt': 'smt', 'canyn': 'cyn', 'ldg': 'ldg', 'harbr': 'hbr', 'rest': 'rst', 'shoars': 'shrs',
            'vist': 'vis', 'gdn': 'gdn', 'islnds': 'iss', 'hills': 'hls', 'cresent': 'cres', 'point': 'pt', 'lake': 'lk', 'vlly': 'vly',
            'strav': 'stra', 'crossroad': 'xrd', 'bnd': 'bnd', 'strave': 'stra', 'stravn': 'stra', 'knol': 'knl', 'vlgs': 'vlgs',
            'forge': 'frg', 'cntr': 'ctr', 'cape': 'cpe', 'height': 'hts', 'lck': 'lck', 'highwy': 'hwy', 'trnpk': 'tpke', 'rpd': 'rpd',
            'boulv': 'blvd', 'circles': 'cirs', 'valleys': 'vlys', 'vst': 'vis', 'creek': 'crk', 'mall': 'mall', 'spring': 'spg',
            'brg': 'brg', 'holws': 'holw', 'lf': 'lf', 'est': 'est', 'xing': 'xing', 'trace': 'trce', 'bottom': 'btm',
            'streme': 'should_append = Falsen', 'extensions': 'exts', 'pkwys': 'pkwy', 'islands': 'iss', 'road': 'rd', 'shrs': 'shrs',
            'roads': 'rds', 'glens': 'glns', 'springs': 'spgs', 'missn': 'msn', 'ridge': 'rdg', 'arcade': 'arc',
            'bayou': 'byu', 'crsnt': 'cres', 'junctn': 'jct', 'way': 'way', 'valley': 'vly', 'fork': 'frk',
            'mountains': 'mtns', 'bottm': 'btm', 'forg': 'frg', 'ht': 'hts', 'ford': 'frd', 'hl': 'hl',
            'grdn': 'gdn', 'fort': 'ft', 'traces': 'trce', 'cnyn': 'cyn', 'cir': 'cir', 'un': 'un', 'mtn': 'mtn',
            'flats': 'flts', 'anex': 'anx', 'gatway': 'gtwy', 'rapids': 'rpds', 'villiage': 'vlg', 'flds': 'flds',
            'coves': 'cvs', 'rvr': 'riv', 'av': 'ave', 'pikes': 'pike', 'grv': 'grv', 'vista': 'vis', 'pnes': 'pnes',
            'forests': 'frst', 'field': 'fld', 'branch': 'br', 'grn': 'grn', 'dale': 'dl', 'rds': 'rds', 'annex': 'anx',
            'sqr': 'sq', 'cove': 'cv', 'squ': 'sq', 'skyway': 'skwy', 'ridges': 'rdgs', 'hwy': 'hwy', 'tunnl': 'tunl',
            'underpass': 'upas', 'cliff': 'clf', 'lane': 'ln', 'land': 'land', 'bch': 'bch', 'dvd': 'dv', 'curve': 'curv',
            'cpe': 'cpe', 'summit': 'smt', 'gardens': 'gdns'}



def main():
    while True:
        user_input = input("enter 1 for matching a single pair and 2 for parsing from spreadsheet")
        if user_input == "1":
            single_test()
            break
        elif user_input == "2":
            multi_test()
            break
        else:
            print("Dekh ke enter kar bro")



def strategy(row1, row2):
    
        addr1={}
        addr2={}

        address1 = row1
        address2 = row2

        # address1 = " 21 22nd Ave Central St; Miami;FL;;33142"  
        string1 = address1.split(",")


        addr1["street"] = string1[0].strip()+" "+string1[1].strip()+" "+string1[2].strip()
        addr1["city"] = string1[3].strip()
        addr1["state"] = string1[4].strip()
        addr1["country"] = string1[5].strip()
        addr1["pincode"] = string1[6].strip()




        # address2 = "22nd Avenue, Northwest,  Street;DL;DL;IN;111005"
        string2 = address2.split(",")


        addr2["street"] = string2[0].strip()+" "+string2[1].strip()+" "+string2[2].strip()
        addr2["city"] = string2[3].strip()
        addr2["state"] = string2[4].strip()
        addr2["country"] = string2[5].strip()
        addr2["pincode"] = string2[6].strip()


        points = {
            "street": 0,
            "city": 0.3,
            "state": 0.05,
            "country": 0.05,
            "pincode": 0.6
        }




        street1 = addr1["street"].lower().replace(",", "")
        street2 = addr2["street"].lower().replace(",", "")

        master_keywords=[]
        numbers1 = []
        keywords1 = []
        numbers2 = []
        keywords2 = []
        final_street_score = 0
        score = 0
        street_weight = 0.7
        non_street_weight = 1-street_weight

        if not addr1["pincode"] or not addr2["pincode"]:
            street_weight = 0.9
            non_street_weight = 0.1
            # sum_of_points = points['city'] + points['state']+points['country']
            # print("HAHAHAHAHAHAH")
            # revised_pincode_weight = 0.2
            # diff = points['pincode'] - revised_pincode_weight
            
            # points["pincode"] = revised_pincode_weight
            # points['city'] = points["city"] + (points["city"]/sum_of_points)*diff
            # points['state'] = points["state"] + (points['state']/sum_of_points)*diff
            # points['country'] = points['country'] + (points['country']/sum_of_points)*diff




        suffixDictitems = suffixDict()
        for key, value in suffixDictitems.items():
            master_keywords.append(key)
            master_keywords.append(value)
        

        array_string = ""
        number_found=False
        for (index,i) in enumerate(address1):


            if i.isdigit():
                number_found = True
                array_string=array_string+i
                if index == len(address1)-1:
                    numbers1.append(array_string)
            else:
                if number_found:
                    numbers1.append(array_string)
                    array_string=""
                    number_found = False


        components = street1.split(" ")
        for count, word in enumerate(components):
            for keyword in master_keywords:
                if textdistance.levenshtein(word, keyword) <= 1:
                    keywords1.append(components[count-1])
                    break
        

        print(numbers1)

        array_string = ""
        number_found=False
        for (index,i) in enumerate(address2):


            if i.isdigit():
                number_found = True
                array_string=array_string+i
                if index == len(address2)-1:
                    numbers2.append(array_string)
            else:
                if number_found:
                    numbers2.append(array_string)
                    array_string=""
                    number_found = False


        components = street2.split(" ")
        for count, word in enumerate(components):
            for keyword in master_keywords:
                if textdistance.levenshtein(word, keyword) <= 1:
                    keywords2.append(components[count-1])
                    break
        

        print(numbers2) 


        # if one complete match
        street_score_number = 0
        similar_numbers = {}
        l=0
        n=0
        m=0
        if not (len(numbers1) == 0 or len(numbers2) == 0):

            for i in numbers1:
                temp_numbers = []
                for j in numbers2:

                    if(textdistance.levenshtein(i,j) == 0):
                        temp_numbers.append(1)                        
                        
                    elif (textdistance.levenshtein(i,j) == 1):
                        temp_numbers.append(0.5)    

                if not (i in similar_numbers.keys() and similar_numbers[i] == 1) and temp_numbers:
                    l+=1
                    similar_numbers[i] = max(temp_numbers)

            print(similar_numbers)
            print("HEHE")
            print("street score number", np.array(list(similar_numbers.values())).mean())
            print(l)
            street_score_number= 0 if math.isnan(np.array(list(similar_numbers.values())).mean()) else np.array(list(similar_numbers.values())).mean()
            street_score_number = street_score_number * (l/max(len(numbers1), len(numbers2)))
            print(street_score_number)

            # if len(similar_numbers) == min(len(numbers1), len(numbers2)):
            #     print("match!!")


            # if similar_numbers == numbers1 or similar_numbers == numbers2:
            #     print("matching numbers found", similar_numbers)
            # else:
            #     print("numbers don't match")
        else:
            # levenshtein match on entire string
            print("numbers don't match")


        similar_keywords = {}
        street_score_keyword = 0
        l=0

        print("keywords 1: ", keywords1)
        print("keywords 2: ", keywords2)

        if not (len(keywords2) == 0 or len(keywords1) == 0):

            for i in keywords1:
                temp_keywords = []
                for j in keywords2:
                    temp_keywords.append(1-textdistance.levenshtein.normalized_distance(i,j))    
            
                if (i not in similar_keywords.keys()):
                    l+=1
                    similar_keywords[i] = max(temp_keywords)
            
            if similar_keywords:
                street_score_keyword=0 if math.isnan(np.array(list(similar_keywords.values())).mean()) else np.array(list(similar_keywords.values())).mean()
                street_score_keyword = street_score_keyword * (l/max(len(keywords1), len(keywords2)))
            else:
                street_score_keyword = 0
            # print(street_score_keyword)

            # if len(similar_numbers) == min(len(numbers1), len(numbers2)):
            #     print("match!!")
        else:
            street_score_keyword = 0
            
        if not numbers1 and not numbers2:
            number_weight = 0
        else:
            number_weight = 0.3

        if not keywords1 and not keywords2:
            keyword_weight = 0
        else:
            keyword_weight = 0.2
        
                    
        street_score_token = 0
        token_weight = 0.5

        print(max(len(numbers1),len(numbers2)))

        street_score_token = 1-textdistance.jaccard.normalized_distance(addr1['street'], addr2['street'])

        if (not numbers1 and not numbers2) and (not keywords1 and not keywords2):
            token_weight = 1
        else:
            token_weight = 1 - 0.6 * (((len(similar_numbers)/max(len(numbers1),len(numbers2),1)) + (len(similar_keywords)/max(len(keywords1),len(keywords2),1)))/2)

        number_weight = 0.75*(1-token_weight)
        keyword_weight = 0.25*(1-token_weight)    
        print(street_score_keyword)

        final_street_score = street_score_number*number_weight + street_score_keyword*keyword_weight + street_score_token*token_weight
        print(street_score_keyword)
        print("final street score: ", final_street_score)

        score = 0
        

        for key in addr1:
            if addr1[key] == "" or addr2[key] == "":
                continue
            if key == "pincode":
                if addr1[key] in addr2[key] or addr2[key] in addr1[key]:
                    score = score + points[key]
                else:
                    score = score + (points[key]*(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key])))
            else:
                score = score + (points[key]*(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key])))


        print("non street score: ", score)
        total_score = (street_weight*final_street_score + non_street_weight*score)*100
        print("total score: ", total_score)

        jaccard_score = (1-textdistance.jaccard.normalized_distance(address1, address2))*100

        return [total_score, jaccard_score]


def single_test():

    df = pd.read_excel('../Samples.ods', sheet_name="Sheet3")

    address1 = df.Bad[0]
    address2 = df.Original[0]

    final_score = strategy(address1, address2)
    print("final_score  ", final_score[0])

def multi_test():
    df = pd.read_excel('../Samples.ods', sheet_name="Sheet2")

    final_scores = []
    jaccard_scores = []
    for row1, row2 in zip(df.Bad, df.Original):
        results = strategy(row1, row2)
        final_scores.append(results[0])
        jaccard_scores.append(results[1])
        

    print(final_scores)

    df["scores"] = final_scores
    df["jaccard_scores"] = jaccard_scores

    df.to_excel("../Scores.ods", sheet_name="Sheet1")

    



main()























