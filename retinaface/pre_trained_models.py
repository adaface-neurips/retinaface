from collections import namedtuple

from torch.utils import model_zoo

from retinaface.predict_single import Model

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

def get_model(model_name: str, max_size: int, device: str = "cpu") -> Model:
    model = models[model_name].model(max_size=max_size, device=device)
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[7:]] = state_dict.pop(key)
            
    model.load_state_dict(state_dict)

    return model
