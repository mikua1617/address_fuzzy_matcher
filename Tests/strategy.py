
import textdistance
from textdistance import algorithms

street1 = ""
street2 = ""
addr1={}
addr2={}
scoring_system = 2
algos = ["levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler", "monge_elkan", "hamming", "mlipns", "strcmp95", "smith_waterman", "jaccard", "needleman_wunsch", "overlap", "tversky"]

address1 = "4/5190 Gali No. 4 Krishna Nagar,,,Karol Bagh,IN-DL,IN,"
string1 = address1.split(",")



addr1["streetI"] = string1[0].strip()
addr1["streetII"] = string1[1].strip()
addr1["streetIII"] = string1[2].strip()
addr1["city"] = string1[3].strip()
addr1["state"] = string1[4].strip()
addr1["country"] = string1[5].strip()
addr1["pincode"] = string1[6].strip()



# print(addr1)

address2 = "4/5190, Gali No. 4, Krishna Nagar,Karol Bagh,DL,IN,111005"
string2 = address2.split(",")


addr2["streetI"] = string2[0].strip()
addr2["streetII"] = string2[1].strip()
addr2["streetIII"] = string2[2].strip()
addr2["city"] = string2[3].strip()
addr2["state"] = string2[4].strip()
addr2["country"] = string2[5].strip()
addr2["pincode"] = string2[6].strip()



# print(addr2)



points = {
    "streetI": 0.2,
    "streetII": 0.2,
    "streetIII": 0.1, 
    "city": 0.2,
    "state": 0.15,
    "country": 0.05,
    "pincode": 0.1
}

points2 = {
    "streetI": 0.2,
    "streetII": 0.1,
    "streetIII": 0.1, 
    "streetIIxIII": 0.05,
    "streetIIIxII": 0.05,
    "city": 0.2,
    "state": 0.15,
    "country": 0.05,
    "pincode": 0.1
}


for algo in algos:
    score = 0



    if scoring_system == 1:
        for key in addr1:
            if addr1[key] == "" or addr2[key] == "":
                continue
            if key == "city" and any(char.isdigit() for char in addr1["city"]):
                score = score + (points[key]*(1-eval("textdistance."+algo).normalized_distance(addr1["city"], addr2["pincode"])))
            else:
                score = score + (points[key]*(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key])))
                # print(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key]))    
        

    else:
        
        for key in addr1:
            if addr1[key] == "" or addr2[key] == "":
                continue
            
            score = score + (points2[key]*(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key])))
            # print(1-eval("textdistance."+algo).normalized_distance(addr1[key], addr2[key]))

        score = score + points2["streetIIxIII"]*(1-eval("textdistance."+algo).normalized_distance(addr1["streetII"], addr2["streetIII"]))
        score = score + points2["streetIIIxII"]*(1-eval("textdistance."+algo).normalized_distance(addr1["streetIII"], addr2["streetII"]))

    # print("score: ", score)
    print(score, end ="\t")

# if textdistance.monge_elkan(addr1["pincode"], addr2["pincode"]) >0.8:
#     textdistance.monge_elkan(addr1["state"], addr2["state"])

# else:
#     if textdistance.monge_elkan(addr1["state"], addr2["state"])>0.9:
#         if textdistance.monge_elkan(addr1["city"], addr2["city"])>0.9:
#             print("pincode probably wrong")
#         else: print("hmmm")   
#     elif textdistance.monge_elkan(addr1["city"], addr2["city"])<0.8:
#         print("Poor match: ", score)
#     else: print("Poor match: ", score)


