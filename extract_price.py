from Market import Market
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filename',dest='filename')
parser.add_argument('-log_dir',dest='log_dir')
result = parser.parse_args()
filename = result.filename
LOG_DIR = result.log_dir

filename = filename
size=0.4
train_size=0.6
test_size=0.3
valid_size=0.1

env = Market(filename,size,train_size,test_size,valid_size)

with open(LOG_DIR+"/train_price.txt",'w') as f:
    f.write(str(list(env.train_close)))

with open(LOG_DIR+"/test_price.txt",'w') as f:
    f.write(str(list(env.test_close)))
