import glob
import importlib
import logging
import re

import h5py
from torch.autograd import Variable

PROJECT="videotexture"

def to_variable(X, cuda=True):
    if isinstance(X, (tuple, list)):
        return tuple(to_variable(x) for x in X)
    else:
        X = Variable(X)
        if cuda:
            return X.cuda().requires_grad_()
        return X.requires_grad_()

def setup_logging(name=None):
    if name:
        name = f"{PROJECT}.{name}"
    else:
        name = PROJECT

    logfile = logging.FileHandler(filename=f"{name}.log", mode='w')
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter('[%(asctime)s, %(levelname)s @%(name)s] %(message)s'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
    logging.basicConfig(level=logging.DEBUG, handlers=[logfile, console])
    return logging.getLogger(name)

def latest_file(globstr):
    fn_candidates = sorted(glob.glob(globstr), reverse=True)
    if len(fn_candidates) == 0:
        raise RuntimeError(f"No files match the pattern {globstr}")
    return fn_candidates[0]

def DynamicLoad(pkg):
    logger = setup_logging("DynamicLoad")
    # Used to dynamically load modules in commandline options.

    def _split_name(modnamefull):
        m = re.fullmatch('(\w+)(\[((\w+(=[\w\.\+\-~@/_\*]+)?,?)*)\])?', modnamefull)
        if m is None:
            logger.error(f"Incorrect import string {modnamefull}")
            raise RuntimeError("Incorrect import.")

        modname = m.group(1)
        props = m.group(3)
        if props is not None:
            # Convert it to a dictionary:
            props = (v.split('=', 1) for v in props.split(","))
            props = {v[0]: (v[1] if len(v) == 2 else True) for v in props}
        else:
            props = {}

        return modname, props

    def _load_actual(modnamefull):
        modname, props = _split_name(modnamefull)
        try:
            load_mod = importlib.import_module(pkg + "." + modname, package="")
        except Exception as e:
            logger.exception(e)
            raise

        try:
            logger.debug("Configuring {}.{} with options: {}".format(pkg, modname, str(props)))
            if hasattr(load_mod, 'build'):
                return load_mod.build(props)
            elif hasattr(load_mod, 'configure'):
                load_mod.configure(props)
                return load_mod
            elif len(props) == 0:
                return load_mod
            else:
                raise RuntimeError('Options passed to item that does not support options.')

        except Exception as e:
            logger.exception(e)
            raise e

    return _load_actual

def loadDataFile(fn):
    h5f = h5py.File(fn)
    seq = []

    in_seq = h5f['seq']
    if isinstance(in_seq, h5py.Dataset):
        seq = in_seq
    else:
        for v in in_seq.values():
            seq.append(v[()])

    params = None
    if 'param' in h5f:
        params = h5f['param'][()]
    return (seq, params)
