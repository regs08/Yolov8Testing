import ast

def load_labels(filename):
    with open(filename, 'r') as file:
        data = file.read()
        labels_dict = ast.literal_eval(data)
    return labels_dict