
_first_run = None
_history_attentions = None
_attention_idx = None
_multiframe = False

def set_first_run(value):
    global _first_run
    _first_run = value

def get_first_run():
    global _first_run
    return _first_run

def set_history_attentions(value):
    global _history_attentions
    _history_attentions = value
    
def del_history_attentions():
    global _history_attentions
    del _history_attentions
    
def append_history_attentions(value):
    global _history_attentions
    _history_attentions.append(value)

def get_history_attentions_with_idx(idx):
    global _history_attentions
    return _history_attentions[idx]

def len_history_attentions():
    global _history_attentions
    return len(_history_attentions)

def set_history_attentions_with_idx(value, idx):    
    global _history_attentions
    _history_attentions[idx] = value
    


def set_attention_idx(value):
    global _attention_idx
    _attention_idx = value

def get_attention_idx():
    global _attention_idx
    return _attention_idx

def set_multiframe(value):
    global _multiframe
    _multiframe = value

def get_multiframe():
    global _multiframe
    return _multiframe



#for sparse attention
def set_history_attentions_sparse(value):
    global _history_attentions_sparse
    _history_attentions_sparse = value
    
def append_history_attentions_sparse(value):
    global _history_attentions_sparse
    _history_attentions_sparse.append(value)

def get_history_attentions_sparse_with_idx(idx):
    global _history_attentions_sparse
    return _history_attentions_sparse[idx]

def set_history_attentions_sparse_with_idx(value, idx):
    global _history_attentions_sparse
    _history_attentions_sparse[idx] = value

def del_history_attentions_sparse():
    global _history_attentions_sparse
    del _history_attentions_sparse

def set_attention_idx_sparse(value):
    global _attention_idx_sparse
    _attention_idx_sparse = value

def get_attention_idx_sparse():
    global _attention_idx_sparse
    return _attention_idx_sparse