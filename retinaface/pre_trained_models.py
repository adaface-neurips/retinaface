from collections import namedtuple
import torch
import os, sys
from torch.utils import model_zoo
from torch.hub import get_dir, _is_legacy_zip_format, download_url_to_file, HASH_REGEX
from typing import Any, Dict, Optional
from retinaface.predict_single import Model
from torch.serialization import MAP_LOCATION
import warnings
from urllib.parse import urlparse  # noqa: F401
import zipfile
from typing_extensions import deprecated


model = namedtuple("model", ["url", "model"])

models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",  # noqa: E501 pylint: disable=C0301
        model=Model,
    ),
    "biubug6": model(
        url="https://github.com/adaface-neurips/retinaface/raw/master/models/Resnet50_Final.zip?raw=true",
        model=Model,
    )
}


def unzip_and_rename(
    filename: str,
    model_dir: str,
) -> Dict[str, Any]:
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)

    os.rename(extracted_file, filename)

def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> state_dict = torch.hub.load_state_dict_from_url(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        ... )

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        unzip_and_rename(cached_file, model_dir)
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)

def get_model(model_name: str, max_size: int, device: str = "cpu") -> Model:
    model = models[model_name].model(max_size=max_size, device=device)
    state_dict = load_state_dict_from_url(models[model_name].url, file_name = model_name + ".pth",
                                          progress=True, map_location="cpu")
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)
            
    model.load_state_dict(state_dict)

    return model
