import functools
import importlib
import os
from functools import partial
from inspect import isfunction

import fsspec
import torch
from einops import repeat


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """

    return self


def get_string_from_tuple(s):
    try:
        # check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # convert the string to a tuple
            t = eval(s)
            # check if the type of t is tuple
            if isinstance(t, tuple):
                return t[0]
            else:
                pass
    except:
        pass
    return s


def is_power_of_two(n):
    """
    Return True if n is a power of 2, otherwise return False.
    """

    if n <= 0:
        return False
    else:
        return (n & (n - 1)) == 0


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
                enabled=enabled,
                dtype=torch.get_autocast_gpu_dtype(),
                cache_enabled=torch.is_autocast_cache_enabled()
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def repeat_as_img_seq(x, num_frames):
    if x is not None:
        if isinstance(x, list):
            new_x = list()
            for item_x in x:
                new_x += [item_x] * num_frames
            return new_x
        else:
            x = x.unsqueeze(1)
            x = repeat(x, "b 1 ... -> (b t) ...", t=num_frames)
            return x
    else:
        return None


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    else:
        return path


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return x.ndim == 2


def isneighbors(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        else:
            raise KeyError("Expected key `target` to instantiate")
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return torch.cat((x, x.new_zeros([1])))


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def get_configs_path() -> str:
    """
    Get the `configs` directory.
    """

    this_dir = os.path.dirname(__file__)
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs")
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find configs in {candidates}")
