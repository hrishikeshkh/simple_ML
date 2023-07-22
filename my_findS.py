import pandas as pd

df = pd.read_csv("enjoysport.csv")

print(df)
 
a = df.values.tolist()

hypo = []

for i in a:
    if i[-1] == "Yes":
        if hypo == []:
            hypo = i[:-1]
        else:
            for j in range(len(i) - 1):
                if i[j] != hypo[j]:
                    hypo[j] = "?"
    print("for the ", a.index(i) + 1,"th training instance is ", hypo)

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypo)