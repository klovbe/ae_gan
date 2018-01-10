
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import codecs
from sklearn.utils.extmath import softmax

# train_path = r"/home/bigdata/cwl/data_preprocessed/reprocess.csv"
# infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"

def reverse_normalization(data, factor=1e6, **kwargs):
  # data[ data >= 13.5 ] = 13.5
  exp = np.expm1(data)
  row_sum = np.sum(exp, axis=1)
  row_sum = np.expand_dims(row_sum, 1)
  div = np.divide(exp, row_sum)
  div = np.log(1 + factor * div)
  return div

def row_normalization(data, factor=1e6, **kwargs):
  row_sum = np.sum(data, axis=1)
  row_sum = np.expand_dims(row_sum, 1)

  div = np.divide(data, row_sum)
  print("begin to loop cal log...")
  m, n = np.shape(div)
  div = np.log(1 + factor * div)
  print("data range is {}".format(np.max(div),np.min(div)))
  return div

def divide_max(data):
  matrix_max = np.max(data)
  trans = np.divide(data, matrix_max)
  return trans

def log(data):
  return np.log(data + 1.0)

def same(data):
  return data

trans_map = {
  "reverse":reverse_normalization,
  "row_normal":row_normalization,
  "div_max": divide_max,
  "same": same,
  "log":log,
  "log10": log10
}

def sub_handle(path, way, header=0, ind_col=0, save_path=None,**kwargs):
  data = pd.read_csv(path, header=header, sep=",", index_col=ind_col)
  print("read from {} done".format(path))
  [m,n] = data.shape
  if m>n:
    data = data.transpose()
  print("{} data_shape is {}".format(path, data.shape))

  print("origin gene nums: {}".format(len(data.columns)))
  if kwargs.get("filter_gene") is not None:
    x = kwargs.get("filter_gene")
    col_list = []
    for col in data.columns:
      if (data[col] > 0.0).sum() >= x:
        col_list.append(col)
    data = data[col_list]
  print("now gene nums: {}".format(len(data.columns)))
  columns = list(data.columns)
  data = data.values
  data = trans_map[way](data,**kwargs)
  print("data range is {}".format(np.max(data), np.min(data)))
  data = pd.DataFrame(data, columns=columns)

  if save_path is not None:
    data.to_csv(save_path, index=False)
    print("saved to {}".format(save_path))
  return data

def handle_data(train_path, test_path, save_train_path, save_test_path, way = "div_max", **kwargs):
  train_df = sub_handle(train_path, way, **kwargs)
  test_df = sub_handle(test_path, way, **kwargs)

  all_df = pd.concat([train_df, test_df], axis=0)

  all_df.to_csv(save_train_path, index=False)
  print("save to {}".format(save_train_path))
  test_df.to_csv(save_test_path, index=False)
  print("save to {}".format(save_test_path))

if __name__ == "__main__":

  # train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop80.csv"
  # infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"
  #
  # handle_data(train_path, infer_path, r"/home/bigdata/cwl/Gan/data/drop80_log.train", r"/home/bigdata/cwl/Gan/data/drop80_log.infer", way="log")

  # train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop60.csv"
  # infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop60.csv"
  #
  # handle_data(train_path, infer_path, r"/home/bigdata/cwl/Gan/data/drop60_log.train",
  #             r"/home/bigdata/cwl/Gan/data/drop60_log.infer", way="log")

  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_kolod.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/cluster/h_kolod.train",factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_usoskin.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/cluster/h_usoskin.train",factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_pollen.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/cluster/h_pollen.train",factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_usoskin_fgene.csv", "row_normal",save_path="/home/bigdata/cwl/Gan/data/cluster/h_usoskin_fgene.train", factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_brain_fgene.csv", "row_normal",save_path="/home/bigdata/cwl/Gan/data/cluster/h_brain_fgene.train", factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/cluster/h_brain.csv", "row_normal",save_path="/home/bigdata/cwl/Gan/data/cluster/h_brain.train", factor=1e6)

  # sub_handle("/home/bigdata/cwl/Gan/chu/chu_sc_handle.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/chu/chu_sc_handle.train",factor=1e6)
  # sub_handle("/home/bigdata/cwl/Gan/chu/chu_h1_handle.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/chu/chu_h1_handle.train",factor=1e6)



  # sub_handle("/home/bigdata/cwl/Gan/cluster/kolod.csv", "same", save_path="/home/bigdata/cwl/Gan/data/cluster/kolod.train")
  # sub_handle("/home/bigdata/cwl/Gan/cluster/usoskin.csv", "same", save_path="/home/bigdata/cwl/Gan/data/cluster/usoskin.train")
  # sub_handle("/home/bigdata/cwl/Gan/cluster/pollen.csv", "same", save_path="/home/bigdata/cwl/Gan/data/cluster/pollen.train")

  # sub_handle("F:/project/simulation_data/drop60_p.csv", "row_normal",
  #            save_path="F:/project/simulation_data/drop60_p.train", factor=1e6)

  sub_handle("F:/project/glcic_mine/data/h_brain.csv", "row_normal", save_path="F:/project/simulation_data/h_brain.train",factor=1e6)
  sub_handle("F:/project/glcic_mine/data/h_kolod.csv", "row_normal",
             save_path="F:/project/simulation_data/h_kolod.train", factor=1e6)
  sub_handle("F:/project/glcic_mine/data/h_pollen.csv", "row_normal",
             save_path="F:/project/simulation_data/h_pollen.train", factor=1e6)
  sub_handle("F:/project/glcic_mine/data/h_usoskin.csv", "row_normal",
             save_path="F:/project/simulation_data/h_usoskin.train", factor=1e6)