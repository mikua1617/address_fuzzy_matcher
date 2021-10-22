from numpy.lib.function_base import average
import textdistance
import numpy as np
import pandas as pd
import math

algo="levenshtein"


def suffixDict():
    """
    List of common US address abbreviations to match address keywords. Key value pairs later converted into list
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
    """
    Main function that takes user input at the beginning of the program. 1 takes a single input (a pair of addresses) from a spreadsheet
    while 2 parses pairs of addresses from a sheet until it reaches EOF.

    All addresses need to have exactly 6 commas that delineate the 7 parts of an address - Street line 1, Street line 2, Street line 3, City, State,
    Country and Pincode. If one of the fields is not available for the address then any amount of whitespace or consecutive commas will work fine
    """

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
    
        """
        Entire logic for matching resides in this function

        row1 and row2 are comma separated addresses sourced directly from spreadsheets. Here both addresses are fragmented according to the delimiter
        that is specified as a comma
        """

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


        """
        points is the weightage assigned to each component while generating the overall score. This has a 0 score for street because street
        scoring is done separately
        """

        points = {
            "street": 0,
            "city": 0.3,
            "state": 0.05,
            "country": 0.05,
            "pincode": 0.6
        }


        """
        Removing all commas withing address fragments and converting all characters to lower case.
        """


        street1 = addr1["street"].lower().replace(",", "")
        street2 = addr2["street"].lower().replace(",", "")

        """
        Variable declaration for a particular matching run
        """

        master_keywords=[]
        numbers1 = []
        keywords1 = []
        numbers2 = []
        keywords2 = []
        final_street_score = 0
        score = 0
        street_weight = 0.7
        non_street_weight = 1-street_weight


        """
        The matching process is divided into 2 parts. One for the street lines part of an address and one for the city state country and pincode.
        Each part has a final weightage in the final score which is determined by street_weight and non_street_weight respectively
        """


        if not addr1["pincode"] or not addr2["pincode"]:

            """
            If either of the addresses does not contain a pincode, then the pincode weightage is reduced to 0.2 and the remainder is 
            equally divided over city state and country in the ratio of their existing weights.

            i.e. 0.6 weight for pincode becomes 0.2 and 0.4 is divided among city, state and country in the ratio of 0.3:0.05:0.05. 
            This changes the ratios for city to 0.3+(0.3/(0.3+0.05+0.05))*0.4 = 0.6.
            Similarly the weights for state and country change too to make the final weights 0.6,0.1,0.1,0.2 for city state country and pincode respectively

            This is needed because if an address doesnt contain a pincode then the score suffers a lot despite the addresses being similar since
            pincode has a high weightage 
            """

            # street_weight = 0.9
            # non_street_weight = 0.1
            sum_of_points = points['city'] + points['state']+points['country']
            # print("HAHAHAHAHAHAH")
            revised_pincode_weight = 0.2
            diff = points['pincode'] - revised_pincode_weight
            
            points["pincode"] = revised_pincode_weight
            points['city'] = points["city"] + (points["city"]/sum_of_points)*diff
            points['state'] = points["state"] + (points['state']/sum_of_points)*diff
            points['country'] = points['country'] + (points['country']/sum_of_points)*diff


        """
        Converting the dictionary of key value pairs of address suffixes to a list master_keywords for easier parsing
        """

        suffixDictitems = suffixDict()
        for key, value in suffixDictitems.items():
            master_keywords.append(key)
            master_keywords.append(value)
        


        """
        This fragment loops through each address and captures instances of numbers and keywords. 
        
        """


        array_string = ""
        number_found=False
        for (index,i) in enumerate(street1):
            """
            Go through each character and if it is a number, then add it to the list of numbers. All consecutive occurences of a number
            count as one number in the final list of numbers. numbers1 is the list of all numbers that occur in the address 1
            """

            if i.isdigit():
                number_found = True
                array_string=array_string+i
                if index == len(street1)-1:
                    numbers1.append(array_string)
            else:
                if number_found:
                    numbers1.append(array_string)
                    array_string=""
                    number_found = False

        """
        Parse through the street line (all 3) and split all parts according to a comma delimiter. Then check if a word exists in the 
        master keywords list. If it does, then add the word previous to the keyword into the keywords1 list for comparison (Riverwood drive
        would have riverwood as the keyword for comparison)
        """

        components = street1.split(" ")
        for count, word in enumerate(components):
            if word in master_keywords:
                keywords1.append(components[count-1])
            # for keyword in master_keywords:
            #     if textdistance.levenshtein(word, keyword) <= 1:
            #         keywords1.append(components[count-1])
            #         break
        

        print(numbers1)

        """
        Same process as Address1 for Address 2
        """

        array_string = ""
        number_found=False
        for (index,i) in enumerate(street2):


            if i.isdigit():
                number_found = True
                array_string=array_string+i
                if index == len(street2)-1:
                    numbers2.append(array_string)
            else:
                if number_found:
                    numbers2.append(array_string)
                    array_string=""
                    number_found = False


        components = street2.split(" ")
        for count, word in enumerate(components):
            if word in master_keywords:
                keywords2.append(components[count-1])
            # for keyword in master_keywords:
            #     if textdistance.levenshtein(word, keyword) <= 1:
            #         keywords2.append(components[count-1])
            #         break
        

        print(numbers2) 


        """
        The loop below matches the list of numbers for both addresses.
        """

        
        street_score_number = 0
        similar_numbers = {}
        l=0
        
        #condition for checking if any addresses dont have numbers at all
        if not (len(numbers1) == 0 or len(numbers2) == 0):
            """
            Parse through the list of numbers in address 1. For each number in address 1, parse through each number in address 2. 
            Calculate the levenshtein distance for each pair. If the distance is 0 then append the score as 1 and if distance is 1 then append 0.5 as score
            Higher distances have no impact on final score and are considered not to be matches

            Temp_numbers contains all the numbers and for each number in the first address, the final score is the maximum value of all the matches performed
            with address 2 (max of temp_numbers)

            """
            for i in numbers1:
                temp_numbers = []
                for j in numbers2:

                    if(textdistance.levenshtein(i,j) == 0):
                        temp_numbers.append(1)                        
                        
                    elif (textdistance.levenshtein(i,j) == 1):
                        temp_numbers.append(0.5)    

                """
                only create new entry for a number if it doesnt exist already and there is some score present in temp_numbers after matching for 1 
                number in address 1.

                similar_numbers dictionary contains all the matched numbers according to the algo above
                """
                
                
                if not (i in similar_numbers.keys() and similar_numbers[i] == 1) and temp_numbers:
                    l+=1
                    similar_numbers[i] = max(temp_numbers)

            print(similar_numbers)

            """
            The score for street based on number matching is the mean of all the scores in the similar numbers dictionary
            """
            print("street score number", np.array(list(similar_numbers.values())).mean())
            print(l)
            street_score_number= 0 if math.isnan(np.array(list(similar_numbers.values())).mean()) else np.array(list(similar_numbers.values())).mean()
            
            """
            Street score based on number (street_score_number) is further modified based on the fraction of matches found compared to the numbers present
            in the addresses. For example, if numbers1 = ["1234", "999", "000"], numbers2 = ["4543", 999"] then matching keywords will be ["999": 1]
            Scoring for this will become 1 and despite there being a lot of differences in numbers, the numbers part of the street score will be very high.

            This is normalized by creating a multiplier for the score by dividing the number of matches (given by l) by the maximum length of the 
            numbers list for each address. In the above case, the multiplier will be 1/3 as l=1 and max length of numbers is 3. This ensures that the number
            of matches need to include all numbers for a good match
            """
            
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


        """
        Matching for keywords
        """

        similar_keywords = {}
        street_score_keyword = 0
        l=0

        print("keywords 1: ", keywords1)
        print("keywords 2: ", keywords2)

        if not (len(keywords2) == 0 or len(keywords1) == 0):
            """
            Matching for keywords follows a similar logic to numbers i.e. each keyword in address 1 is matched to each keyword in address 2 and the max
            value of match is kept as the score for the word. The only difference is that here, scores are not discretized as 1 or 0.5 because keyword
            lengths can be very high and the threshold for discrete scoring is hard to find. Instead, the score is a simple levenshtein distance normalized
            and subtracted from 1 to get the match instead of distance out of 1
            """
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
        
        

        """
        In addition to numbers and keyword matching, there is an additional matching score based on the jaccard index. This is a token based 
        matching algorithm that considers words as tokens and nullifies the impact of interchanged words in a string. 

        Jaccard matching is needed in the case of addresses where neither numbers or keywords are present. In such cases, number and keyword scores become
        zero and overall match is poor despite the other parts of address being good matches.


        """

                    
        street_score_token = 0
        token_weight = 0.5

        print(max(len(numbers1),len(numbers2)))

        """
        Calculate the normalized jaccard similarity between the street lines for both addresses
        """

        street_score_token = 1-textdistance.jaccard.normalized_distance(addr1['street'], addr2['street'])
        print("jaccard street",street_score_token)

        """
        The jaccard score also has a variable weight system to combat cases without numbers or keywords. 
        
        The weight for the jaccard score depends inversely on the length of the similar numbers and keywords found as a fraction of the total numbers and
        keywords. This is needed because if the number of similar numbers is close to the number of numbers present in the addresses, then the jaccard score
        doesn't add much value as much of the matching is already done in the number and keyword matching portion. However, if similar numbers and keywords
        are not a significant fraction of the total numbers and keywords in the addresses, then the weight of the jaccard score increases proportionally

        For example, if numbers1 = ["1234", "999", "000"], numbers2 = ["4543", 999"] then matching numbers = ["999"]. keywords1 = ["Raymond", "Whispy"], 
        keywords2 = ["Virginia", "Raymond"] then matching keywords = ["Raymond"]

        Hence the jaccard weight (token weight) = 1-0.9(1/max(3,2)+1/max(2,2))/2) = 1-0.9*(1/3+1/2)/2 = 1-0.9*0.8333/2 = 1-0.25 = 0.625

        If the numbers2 was just ["4543"] then the jaccard score would increase to 1-0.9*(0+1/2)/2 = 0.775 since number matching was not conclusive as there
        could be extra or missing numbers in one address despite being the same address
        """

        
        token_weight = 1 - 0.9 * (((len(similar_numbers)/max(len(numbers1),len(numbers2),1)) + (len(similar_keywords)/max(len(keywords1),len(keywords2),1)))/2)

        """
        Number and keyword weights vary accordingly with token weight with greater weight for numbers
        """
        number_weight = 0
        keyword_weight = 0


        """
        The weights assigned to numbers and keywords are variable. If numbers dont exist in either of the addresses then the weight for numbers is
        removed. Same for keywords.
        """

        if (not numbers1 and not numbers2) and (not keywords1 and not keywords2):
            token_weight = 1
        elif not numbers1 and not numbers2:
            number_weight = 0
            keyword_weight = 1-token_weight
        elif not keywords1 and not keywords2:
            number_weight = 1-token_weight
            keyword_weight = 0
        else:
            number_weight = 0.66*(1-token_weight)
            keyword_weight = 0.34*(1-token_weight)


        
  
        print(street_score_keyword)

        final_street_score = street_score_number*number_weight + street_score_keyword*keyword_weight + street_score_token*token_weight
        print("final street score: ", final_street_score)

        

        """
        For the second part of the match (the non street part including city, state, country and pincode) the matching is done purely by a 
        simple levenshtein distance metric across the 2 addresses. the distance is normalized using the length of the strings and then subtracted
        from 1 to get the match value
        """
        
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

    """
    this method reads Samples.ods and takes the first row from Sheet 3 and calculates the match value for one pair of addresses and prints the 
    output along with a jaccard score of the entire strings compared simultaneously
    """

    df = pd.read_excel('../Samples.ods', sheet_name="Sheet3")

    address1 = df.Bad[0]
    address2 = df.Original[0]

    final_score = strategy(address1, address2)
    print("final_score  ", final_score[0])

def multi_test():

    """
    This method iterates over the values from the file Samples.ods in Sheet2 and calculates the match and jaccard scores for each pair and 
    dumps the values into a file Scores.ods in Sheet1
    """

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























