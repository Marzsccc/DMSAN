import os

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' Directory created successfully.')
        pass
    else:
        print(path + ' Directory already exists.')
        pass


if __name__ == '__main__':
    mkdir('./model_save')
    mkdir('./dataset')
