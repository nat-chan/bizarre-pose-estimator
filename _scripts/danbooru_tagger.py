import sys
import json
from tqdm import tqdm
from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d

from _train.danbooru_tagger.models.kate import Model as DanbooruTagger
model = DanbooruTagger.load_from_checkpoint(
    './_train/danbooru_tagger/runs/waning_kate_vulcan0001/checkpoints/'
    'epoch=0022-val_f2=0.4461-val_loss=0.0766.ckpt'
)

def infer(self, images, return_more=True):
    # images: list of images
    imgs = torch.cat([
        I(img)
            .resize_square(self.hparams.largs.danbooru_sfw.size)
            .alpha_bg(c='w')
            .convert('RGB')
            .tensor()[None,]
        for img in images
    ]).to(self.device)
    self.eval()
    with torch.no_grad():
        out = self.forward(imgs, return_more=False)
    if return_more:
        out['prob_dict'] = [
            {
                r['name']: p
                for r,p in zip(self.rules, x)
            }
            for x in torch.sigmoid(out['raw']).cpu().numpy()
        ]
    return out

if __name__ == '__main__':
    lines = sys.stdin
    for fname, dname in tqdm([line.strip().split() for line in lines]):
        img = I(fname)
        ans = infer(model, [img,])
        data = {k: float(v) for k, v in ans['prob_dict'][0].items()}
        with open(dname, "w") as f:
            json.dump(data, f)

#for k,v in sorted(ans['prob_dict'][0].items(), key=lambda x: -x[1]):
#    if v>=0.5:
#        print(v,k)
#
