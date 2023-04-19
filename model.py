import torch
import ipywidgets as widgets
import requests
from torchvision import models

def get_robust(arch, path):
    backbone = None
    
    try:
        backbone = eval(f'models.{arch}()')
    except:
        raise ValueError('Model Not Found!')

    checkpoint = torch.load(path)
    state_dict_path = 'model'
    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]:v for k,v in sd.items()}
    sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
    backbone.load_state_dict(sd_t)
    return backbone


def download_and_load_backnone(url, model_name):
    arch = '_'.join(model_name.split('_')[:-2])
    print(arch, model_name)
    ckpt_path = os.path.join('./robust_pretrained_models/', f'{model_name}.ckpt')
    
    # Check if checkpoint file already exists
    if os.path.exists(ckpt_path):
        print(f'{model_name} checkpoint file already exists.')
        return get_robust(arch, ckpt_path)

    r = requests.get(url, allow_redirects=True)  # to get content after redirection
    ckpt_url = r.url
    with open(ckpt_path, 'wb') as f:
        f.write(r.content)

    return get_robust(arch, ckpt_path)

robust_urls = {
    'resnet18_linf_eps0.5': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet18_linf_eps1.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet18_linf_eps2.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet18_linf_eps4.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet18_linf_eps8.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',

    'resnet50_linf_eps0.5': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet50_linf_eps1.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet50_linf_eps2.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet50_linf_eps4.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'resnet50_linf_eps8.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    
    'wide_resnet50_2_linf_eps0.5': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps0.5.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'wide_resnet50_2_linf_eps1.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps1.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'wide_resnet50_2_linf_eps2.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps2.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'wide_resnet50_2_linf_eps4.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps4.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
    'wide_resnet50_2_linf_eps8.0': 'https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D',
}

_BACKBONES = {
    "resnet18": lambda pretrained:f"models.resnet18(pretrained={pretrained})",
    "resnet34": lambda pretrained:f"models.resnet50(pretrained={pretrained})",
    "resnet50": lambda pretrained:f"models.resnet50(pretrained={pretrained})",
    "resnet101": lambda pretrained:f"models.resnet101(pretrained={pretrained})",
    "resnet152": lambda pretrained:f"models.resnet152(pretrained={pretrained})",
}

_ROBUST_BACKBONES = {f'Robust_{key}': (lambda k, v: lambda: download_and_load_backnone(v, k))(key, value)  for key, value in robust_urls.items()}

_BACKBONES.update(_ROBUST_BACKBONES)

def load(name, pretrained=False):
    if name.startswith('Robust'):
      return _BACKBONES[name]()
    else:
      return eval(_BACKBONES[name](pretrained))
  
  
#@title Define Model Wrapper
import torch.nn as nn

mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mu = torch.tensor(mu).view(3,1,1).to(device)
std = torch.tensor(std).view(3,1,1).to(device)

class Model(torch.nn.Module):
    def __init__(self, backbone: str, pretrained: bool, num_classes: int):
        super().__init__()
        self.norm = lambda x: ( x - mu ) / std
        self.backbone = load(backbone, pretrained).to(device)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes + 1)

    def forward(self, x):
        x = self.norm(x)
        z = self.backbone(x)
        return z  