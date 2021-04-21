# Images resize to 512 * 512
import glob
import os.path

from PIL import Image  # 提取目录下所有图片,更改尺寸后保存到另一目录


def convertjpg(jpgfile, outdir, width=512, height=512):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("../labelme/pictures/*.jpg"):
    convertjpg(jpgfile, "../labelme/pictures")
