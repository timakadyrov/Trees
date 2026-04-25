import pandas as pd

df = pd.read_csv('postupleni.csv')

# считаем индекс джини
def calculate_gini(labels):
    if len(labels) == 0:
        return 0

    probs = labels.value_counts(normalize=True)
    return 1 - sum(probs ** 2)


# делим данные
def split_dataset(df, feature, threshold):
    left_part = df[df[feature] < threshold]
    right_part = df[df[feature] >= threshold]
    return left_part, right_part

# ищем лучшее разделение
def find_best_split(df):
    best_gain = -1
    best_split = None

    start_gini = calculate_gini(df['Accept'])
    features = df.columns[:-1]

    for feature in features:
        unique_values = sorted(df[feature].unique())

        thresholds = [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]

        for t in thresholds:
            left, right = split_dataset(df, feature, t)

            if len(left) == 0 or len(right) == 0:
                continue

            weighted_gini = (
                (len(left) / len(df)) * calculate_gini(left['Accept']) +
                (len(right) / len(df)) * calculate_gini(right['Accept'])
            )

            gain = start_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_split = {
                    'feature': feature,
                    'threshold': t,
                    'gain': gain
                }

    return best_split


# построение дерева
def build_tree(df, depth=0, max_depth=3):

    if calculate_gini(df['Accept']) == 0:
        return df['Accept'].mode()[0]

    if depth >= max_depth:
        return df['Accept'].mode()[0]

    split = find_best_split(df)

    if split is None or split['gain'] <= 0:
        return df['Accept'].mode()[0]

    left_df, right_df = split_dataset(df, split['feature'], split['threshold'])

    node = {
        'feature': split['feature'],
        'threshold': split['threshold'],
        'gain': split['gain'],
        'left': build_tree(left_df, depth + 1, max_depth),
        'right': build_tree(right_df, depth + 1, max_depth)
    }

    return node


def print_tree(node, indent=""):

    if not isinstance(node, dict):
        print(indent + f"  Ответ: {node}")
        return

    print(indent + f"Если {node['feature']} < {node['threshold']:.2f}")
    print(indent + f"(Gain = {node['gain']:.4f})")

    print(indent + "  Да:")
    print_tree(node['left'], indent + "    ")

    print(indent + "  Нет:")
    print_tree(node['right'], indent + "    ")

tree = build_tree(df)

print("Дерево -_-\n")
print_tree(tree)