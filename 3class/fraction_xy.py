import os
import sys
import numpy as np

def numlines(filename) :
  with open(filename) as fi:
    for cnt, line in enumerate(fi):
      n = cnt
      n += 1
  return n

def select_random_p_from_n(p, n) :
  permutation = np.random.permutation(n)
  k = int(p*n + 0.5)
  firstk = permutation[:k]
  return firstk

def select_random_p(p, filename) :
  n = numlines(filename)
  permutation = np.random.permutation(n)
  k = int(p*n + 0.5)
  firstk = permutation[:k]
  return firstk

def select_from_file(f_in, f_out, selection) :
    fo = open(f_out, "w");
    with open(f_in) as fi:
        for cnt, line in enumerate(fi):
            if(cnt in selection) :
                fo.write(line)
    fo.close()

def select_from_two_files(f1_in, f1_out, f2_in, f2_out, selection) :
    select_from_file(f1_in, f1_out, selection)
    select_from_file(f2_in, f2_out, selection)

def out_name(in_name,p,seed) :
    s = str(seed) if seed >= 0 else ""
    r = str(int(p*100))
    basename = os.path.basename(in_name);
    name, extension = os.path.splitext(basename)
    return(name + "_" + s + "_" + r + extension)

# insist on 3 or 4 arguments
if len(sys.argv) != 4 and len(sys.argv) != 5 :
  print(sys.argv[0], "takes 3 or 4 arguments. Not ", len(sys.argv)-1)
  print("Arguments: file_x file_y fraction [seed]. Example: ",
        sys.argv[0]," training.txt testing.txt 0.3 7")
  sys.exit()

f_x = sys.argv[1]
f_y = sys.argv[2]
p = float(sys.argv[3])
seed = -1
if(len(sys.argv) == 5) :
    seed = int(sys.argv[4])
    np.random.seed(seed)

lines_f_x = numlines(f_x)
lines_f_y = numlines(f_y)
assert lines_f_x == lines_f_y, "f_x,f_y must have same length"
selection = select_random_p_from_n(p, lines_f_x)

f_x_out = out_name(f_x,p,seed)
f_y_out = out_name(f_y,p,seed)

select_from_two_files(f_x, f_x_out, f_y, f_y_out, selection)
