import textdistance

a = "levenshtein"

print(eval("textdistance." + a)("abcd", "avcd"))