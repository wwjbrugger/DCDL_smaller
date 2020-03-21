import wittgenstein as lw
import pandas as pd
from sklearn.model_selection import train_test_split
import model.boolean_Formel as bofo




def wittgenstein_ripper(data, class_feat, max_rules):
    """

    @param data:  label und daten zusammen
    @param class_feat:column name der Label
    @param max_rules: anzahl an regeln
    @return:
    """
    ripper_clf = lw.RIPPER(k = 10, prune_size=0.3)
    ripper_clf.verbosity = 1  # Scale of 1-5
    ripper_clf.fit(data,class_feat=class_feat,  random_state=0,  pos_class= 1,  max_rules=max_rules)
    rule_set = ripper_clf.ruleset_

    test_X = data.drop(class_feat, axis=1)
    test_y = data[class_feat]
    accuracy_score=ripper_clf.score(test_X, test_y) # Default metric is accuracy
    return rule_set, accuracy_score

def np_to_padas(data, label):
    """

    @param data:
    @param label:
    @return: pandas tabel in dem data zusammen geführt sind
    """
    col_name = []
    for i, col in enumerate(data[0]):
        col_name.append('pixel_{}'.format(i))
    df = pd.DataFrame(data,columns= col_name )
    df.insert(0, column='label', value= label)
    return df

if __name__ == '__main__':
    df = pd.read_csv('house-votes-84.csv')

    class_feat = 'Party'
    train, test = train_test_split(df, random_state=0)
    wittgenstein_ripper(train, class_feat)
    #ripper_clf.fit(train, class_feat='Party', random_state=0)
   # ripper_clf.ruleset_ # Access underlying model