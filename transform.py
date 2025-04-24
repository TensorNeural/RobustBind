def unnormalize_inplace(x, mean_t, std_t):
    """
    x is currently in normalized space => (x - mean)/std
    This function brings x back into [0,1] range:
      x = x * std + mean
    """
    x.mul_(std_t).add_(mean_t).clamp_(0, 1)
    return x

def normalize_inplace(x, mean_t, std_t):
    """
    x is currently in [0,1] => we apply x = (x - mean)/std
    to produce the model's expected normalized domain.
    """
    x.sub_(mean_t).div_(std_t)
    return x