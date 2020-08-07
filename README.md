# tch to python poc

## Environment

Check pytorch binary flag

```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Set env. I'm using miniconda.

```bash
export LIBTORCH_CXX11_ABI=0 # based on previous result
export LIBTORCH=$HOME/miniconda3/envs/pt1.6/lib/python3.8/site-packages/torch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

Link the dynamic library

```bash
ln -s target/debug/libtch.so tch.so
```