# SetTime : 2021/9/1 19:38 
# Coding : utf-8 
# Author : marzsccc
# Mail : marzsccc@163.com
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
