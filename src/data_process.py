
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import math

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

def reverse(data, base=math.e ,**kwargs):
  exp = base^data-1
  return np.floor(exp)

def row_normalization(data, base=math.e, factor=1e6, **kwargs):
  row_sum = np.sum(data, axis=1)
  row_sum = np.expand_dims(row_sum, 1)

  div = np.divide(data, row_sum)
  print("begin to loop cal log...")
  m, n = np.shape(div)
  div = np.log(1 + factor * div)/np.log(base)
  print("data range is {}".format(np.max(div),np.min(div)))
  return div

def divide_max(data):
  matrix_max = np.max(data)
  trans = np.divide(data, matrix_max)
  return trans

def log(data,base=math.e):
  return np.log(data + 1.0)/np.log(base)

def same(data):
  return data

trans_map = {
  "reverse":reverse,
  "re_norm":reverse_normalization,
  "row_normal":row_normalization,
  "div_max": divide_max,
  "same": same,
  "log":log,

}

def process_data(path, model_name, input_base=math.e, save_base=10, is_log=True, header=0, ind_col=0,**kwargs):

  data = pd.read_csv(path, header=header, sep=",", index_col=ind_col)
  print("read from {} done".format(path))
  [m,n] = data.shape
  if m>n:
    data = data.transpose()
  print("{} data_shape is {}".format(path, data.shape))

  ori_outdir = "F:/project/simulation_data/origin"+ model_name+".csv"
  ori_logoutdir = "F:/project/simulation_data/log"+ model_name+".csv"

  if is_log:
    data.to_csv(ori_logoutdir,index=False)
    print("saved origin to {}".format(ori_logoutdir))
    dv = data.values
    data = trans_map["reverse"](dv,base=input_base,**kwargs)
    data.to_csv(ori_outdir, index=False)
    print("saved origin reverse raw count to {}".format(ori_outdir))
  else:
    data.to_csv(ori_outdir, index=False)
    print("saved origin to {}".format(ori_outdir))
    dv = data.values
    data_ = trans_map["row_normal"](dv, base=save_base, **kwargs)
    data_.to_csv(ori_logoutdir, index=False)
    print("saved log of origin to{}".format(ori_logoutdir))


#filter cell
  print("origin cell nums: {}".format(len(data)))
  if kwargs.get("filter_cell") is not None:
    x = kwargs.get("filter_cell")
    index_list = []
    for row in data.index:
      if x>1.0:
        if (data.loc[row,:] > 0.0).sum() >= x:
          index_list.append(row)
      else:
        if (data.loc[row,:] > 0.0).sum() >= x:
          index_list.append(row)
    data = data.loc[index_list,:]
  print("now cell nums: {}".format(len(data)))

#filter gene
  print("origin gene nums: {}".format(len(data.columns)))
  if kwargs.get("filter_gene") is not None:
    y = kwargs.get("filter_gene")
    col_list = []
    for col in data.colums:
      if x>1.0:
        if (data.loc[:,col] > 0.0).sum() >= y:
          col_list.append(col)
      else:
        if (data.loc[:,col] > 0.0).sum() >= y:
          col_list.append(col)
    data = data.loc[:,col_list]
  print("now gene nums: {}".format(len(data.columns)))

  if kwargs.get("filter_cell") or kwargs.get("filter_gene") :
    outdir_f = "F:/project/simulation_data/origin_f/"+ model_name+".csv"
    logoutdir_f = "F:/project/simulation_data/log_f"+ model_name+".csv"

    data.to_csv(outdir_f,index=False)
    print("save fitered raw count to {}".format(outdir_f))
    columns = list(data.columns)
    data = data.values
    data = trans_map["row_normal"](data,**kwargs)
    # print("data range is {}".format(np.max(data), np.min(data)))
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(logoutdir_f,index=False)
    print("save filtered log cont to {}".format(logoutdir_f))






if __name__ == "__main__":

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

  # sub_handle("F:/project/glcic_mine/data/h_brain.csv", "row_normal", save_path="F:/project/simulation_data/h_brain.train",factor=1e6)
  # sub_handle("F:/project/glcic_mine/data/h_kolod.csv", "row_normal",
  #            save_path="F:/project/simulation_data/h_kolod.train", factor=1e6)
  # sub_handle("F:/project/glcic_mine/data/h_pollen.csv", "row_normal",
  #            save_path="F:/project/simulation_data/h_pollen.train", factor=1e6)
  # sub_handle("F:/project/glcic_mine/data/h_usoskin.csv", "row_normal",
  #            save_path="F:/project/simulation_data/h_usoskin.train", factor=1e6)

  process_data(path, model_name, input_base=math.e, save_base=10, is_log=True, header=0, ind_col=0, **kwargs)