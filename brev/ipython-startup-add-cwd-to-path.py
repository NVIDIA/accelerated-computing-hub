import os, sys
p = os.getcwd()
if p not in sys.path:
  sys.path.insert(0, p)
