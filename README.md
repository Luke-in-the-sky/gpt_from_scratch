# Build GPT, from scratch

Based on the fantastic lecture serie by AK here https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

The code is slightly altered with
1. I use `@dataclass` decorators, which don't play well with `nn.Module`s out of the box, but I fix it
1. I add `DataSet` and `DataLoader` classes, so I don't have too many global vars laying around
1. no GPU at all (device is cpu only): the mac I have at the moment has no GPU, which forces me to think carefully about optimizing intead of just scaling up the net
1. my own comments, for personal reference
