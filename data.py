import os
import pandas as pd
from sklearn.model_selection import train_test_split

SPAM = 1
HAM = 0
TEST_SIZE = 0.3

FILE_PATHS = [('./spam_2/', SPAM),
              ('./easy_ham/', HAM),
              ('./hard_ham/', HAM)
              ]
SKIP_FILES = ['cmd']


def read_datas():
    result = []
    for path, mail_type in FILE_PATHS:
        file_list = os.listdir(path)
        for f in file_list:
            if f in SKIP_FILES:
                continue
            with open(path + f, 'r') as mail:
                content = []
                for line in mail:
                    content.append(line.decode('latin-1'))
                content = '\n'.join(content)
            result.append({'data': content, 'type': mail_type})
    result = pd.DataFrame(result)
    return result


def get_train_test_split():
    data = read_datas()
    return train_test_split(data['data'], data['type'], test_size=TEST_SIZE, random_state=42)