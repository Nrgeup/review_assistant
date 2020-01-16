import pandas as pd


df = pd.read_csv("./Reviews.csv", encoding='utf-8')
df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本

# print(df.shape) # (568454, 10)
# columns = df.columns.values
# ['Id' 'ProductId' 'UserId' 'ProfileName' 'HelpfulnessNumerator' 'HelpfulnessDenominator' 'Score' 'Time' 'Summary' 'Text']

df["Sentiment"] = df["Score"].apply(lambda score: "positive" if score > 3 else "negative")
df["Usefulness"] = (df["HelpfulnessNumerator"]/df["HelpfulnessDenominator"]).apply(lambda n: "useful" if n > 0.8 else "useless")


def trainTestSplit(X,test_size):
    test_data = X.sample(n=test_size, random_state=0, axis=0)
    train_data = X[~X.index.isin(test_data.index)]
    return train_data, test_data


train, test = trainTestSplit(df, test_size=50000)
# ['Id' 'ProductId' 'UserId' 'ProfileName' 'HelpfulnessNumerator'
#  'HelpfulnessDenominator' 'Score' 'Time' 'Summary' 'Text' 'Sentiment'
#  'Usefulness']
train.to_csv("train_data.csv", index=False)
test.to_csv("test_data.csv", index=False)







