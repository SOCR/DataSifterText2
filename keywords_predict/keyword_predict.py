import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns

from functions import get_features, generate_model, text_clean, split_class


def main():
    consider_text = True # consider text OR features
    num_keywords = 10 # num keywords generated
    drop_list = [0, 2, 4, 5, 6] # useless df features
    raw_data = './stay_length_4000_with_notes.csv' # read from
    outcomes = ['length_of_stay_avg', 'gender', 'religion']
    num_class = 5 # for continuous outcomes
    control = {'remove_number': True, 'lemmatize': True, 'normalize': True, 'stop_words': True, 'gram': 1, 'remove_name': True}
    # TODO: remove name

    # read data frame
    raw_df = pd.read_csv(raw_data)
    df = raw_df.drop(raw_df.columns[drop_list], axis=1)
    # get all categorical columns
    cat_columns = df.select_dtypes(['object']).columns
    # convert all categorical columns to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    # not change text
    df['TEXT'] = raw_df['TEXT']
    # detect outcome column
    outcomes = get_features(outcomes, df)

    keywords = []
    for outcome in outcomes:
        y = np.array(df[outcome])
        if len(np.unique(y)) > num_class:
            split_class(y, num_class)
        else:
            num_class = len(np.unique(y))
        if consider_text:
            name, X = text_clean(np.array(df['TEXT']), control)
            if control['normalize']:
                X = (X - X.mean(axis=0)) / X.std(axis=0)
        else:
            temp_df = df.drop(['TEXT'], axis=1)
            X = np.array(temp_df.drop(outcome, axis=1))
            name= np.array(temp_df.columns)
        model = generate_model(X, y, num_class)
        feature_imp = pd.DataFrame({'Value': model.feature_importance(), 'Feature': name})
        keywords = feature_imp.sort_values(by="Value", ascending=False)[0:num_keywords]

        plt.figure(figsize=(20, 40))
        sns.set(font_scale=5)
        sns.barplot(x="Value", y="Feature", data=keywords)
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(f'lgbm_importances-{outcome}.png')

        print('*' * 30)
        print(f'For outcome {outcome}:')
        keywords = np.array(keywords['Feature'])
        for i in range(min(num_keywords, len(keywords))):
            print(f'{i + 1}th: {keywords[i]}')
        print('*' * 30)


if __name__ == "__main__":
    main()
