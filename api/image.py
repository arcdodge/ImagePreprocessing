import os
from urllib.parse import unquote
from flask import Blueprint, jsonify, abort, make_response, request
from io import BytesIO
from libs.SlideCache import SlideCache
import json
from openslide import OpenSlideError
import base64
import subprocess
from config import FILE_ROOT
from PIL import Image
import cv2
import gc
import openslide
import numpy as np
import abc
from preprocessing import Do_Task, nparray_to_pngbase64str
try:
    from preprocessing_cuda import Do_Task_CUDA
    CUDA_AVAILABLE = True
except ImportError:
    print("CUDA preprocessing module not found. Falling back to CPU.")
    Do_Task_CUDA = None
    CUDA_AVAILABLE = False

image = Blueprint('image', __name__)

class ImageLoader(abc.ABC):
    @abc.abstractmethod
    def Load_Image(self, image_path):
        return NotImplemented

    def Cal_Thumb_W_and_H(self, originW, originH, maxPixel):
        print(originW)
        print(originH)
        print(maxPixel)
        ThumbnailW = maxPixel if originW > originH else originW * int(maxPixel) / int(max(originW, originH))
        ThumbnailH = maxPixel if originH > originW else originH * int(maxPixel) / int(max(originW, originH))
        return (int(ThumbnailW), int(ThumbnailH))

    @abc.abstractmethod
    def Get_Thumbnail(self, image_path,ThumbnailW, ThumbnailH):
        return NotImplemented
    
    @abc.abstractmethod
    def Get_Region(self, image_path, mask_path, x, y, w, h, request):
        return NotImplemented

class Normal_Image(ImageLoader):
    def Load_Image(self, image_path):
        print('****************Normal_Image.Load_Image**********************')
        slide = Image.open(image_path)
        return (slide.size[0],slide.size[1])

    def Get_Thumbnail(self, image_path,ThumbnailW, ThumbnailH):
        size = (ThumbnailW, ThumbnailH)

        try:
            with Image.open(image_path) as im:
                im.thumbnail(size)
                # Convert the thumbnail image to a base64 string
                buffered = BytesIO()
                im.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                return img_base64
        except OSError:
            print("cannot create thumbnail for", infile)

    def Get_Region(self, image_path, mask_path, x, y, w, h, request):
        box = (x, y, x + w, y + h)
        with Image.open(image_path) as im:
            region = im.crop(box)
            buffered = BytesIO()
            region.save(buffered, format="PNG")
            ImageRegion = base64.b64encode(buffered.getvalue()).decode("utf-8")

            MaskRegion = ''
            use_cuda = request.form.get('use_cuda', 'false').lower() == 'true'
            print(f"use_cuda: {use_cuda}")
            print(f"CUDA_AVAILABLE: {CUDA_AVAILABLE}")
            if use_cuda and CUDA_AVAILABLE:
                (ImageRegion,MaskRegion) = Do_Task_CUDA(ImageRegion,MaskRegion,request)
            else:
                (ImageRegion,MaskRegion) = Do_Task(ImageRegion,MaskRegion,request)
            IntegrationRegion = ImageRegion[0]

        return (ImageRegion[0],"null",IntegrationRegion)


class OpenSlide_Image(ImageLoader):
    def Load_Image(self, image_path):
        print('****************OpenSlide_Image.Load_Image**********************')

        if os.path.exists(image_path):
            print(f'檔案存在: {image_path}')
        else:
            print(f'錯誤: 檔案不存在 - {image_path}')


        slide = openslide.OpenSlide(image_path)
        return (slide.dimensions[0],slide.dimensions[1])

    def Get_Thumbnail(self, image_path,ThumbnailW, ThumbnailH):
        slide = openslide.OpenSlide(image_path)
        nparray = np.array(slide.get_thumbnail((ThumbnailW,ThumbnailH)))[:,:,:3]

        img_base64 = nparray_to_pngbase64str(nparray)
        
        return img_base64

    def Get_Region(self, image_path, mask_path, x, y, w, h, request):
        slide = openslide.OpenSlide(image_path)
        back = np.array(slide.read_region((x, y), 0, (w, h)))[:,:,:3]

        ImageRegion = nparray_to_pngbase64str(back)
        MaskRegion = ''
        if MaskRegion == '':
            print("MaskRegion is an empty string.")
        else:
            print(f"MaskRegion is not an empty string.:{MaskRegion}")
        use_cuda = request.form.get('use_cuda', 'false').lower() == 'true'
        
        print(f"use_cuda: {use_cuda}")
        print(f"CUDA_AVAILABLE: {CUDA_AVAILABLE}")
        if use_cuda and CUDA_AVAILABLE:
            (ImageRegion,MaskRegion) = Do_Task_CUDA(ImageRegion,MaskRegion,request)
        else:
            (ImageRegion,MaskRegion) = Do_Task(ImageRegion,MaskRegion,request)
        IntegrationRegion = ImageRegion[0]

        return (ImageRegion[0],"null",IntegrationRegion)

class DCM_Image(ImageLoader):
    def Load_Image(self, image_path):
        print('****************dcmimage.Load_Image**********************')
         
        dcm_path = image_path.rsplit('/', 1)[0]
        slide = openslide.OpenSlide(os.path.join(dcm_path, "SM"+str(0).zfill(6)+".dcm"))
        return (slide.dimensions[0],slide.dimensions[1])

    def Get_Thumbnail(self, image_path,ThumbnailW, ThumbnailH):
        dcm_path = image_path.rsplit('/', 1)[0]
        osr = openslide.OpenSlide(os.path.join(dcm_path, "SM"+str(0).zfill(6)+".dcm"))
        nparray = np.array(osr.get_thumbnail((ThumbnailW,ThumbnailH)))[:,:,:3]

        img_base64 = nparray_to_pngbase64str(nparray)
        
        return img_base64

    def Get_Region(self, image_path, mask_path, x, y, w, h, request):
        dcm_path = image_path.rsplit('/', 1)[0]
        osr = openslide.OpenSlide(os.path.join(dcm_path, "SM"+str(0).zfill(6)+".dcm"))
        scale = 1
        back = np.array(osr.read_region((x, y), 0, (w, h)))[:,:,:3]

        SR = dcmread(mask_path)
        regions = hd.sr.utils.find_content_items(
                dataset=SR,
                value_type=hd.sr.ValueTypeValues.SCOORD3D,
                recursive=True
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        for reg in regions:
            r_np = np.array(reg.GraphicData)
            r_np = r_np.reshape(-1,3)[:,:2]/scale
            r_np[:,0] = r_np[:,0] - x
            r_np[:,1] = (osr.dimensions[1] - r_np[:,1]) - y
                
            r_np = np.expand_dims(r_np, 0).astype(np.int32)

            app = cv2.approxPolyDP(r_np, 3, True)

            mask = cv2.drawContours(mask,[app], -1,255, -1)

        #back為原圖
        #preprocessing
        
        ImageRegion = nparray_to_pngbase64str(back)
        MaskRegion = nparray_to_pngbase64str(mask)
        use_cuda = request.form.get('use_cuda', 'false').lower() == 'true'
        if use_cuda and CUDA_AVAILABLE:
            (ImageRegion,MaskRegion) = Do_Task_CUDA(ImageRegion,MaskRegion,request)
        else:
            (ImageRegion,MaskRegion) = Do_Task(ImageRegion,MaskRegion,request)
        
        # 將 bytes 轉換成 np.array
        img_bytes_data = base64.b64decode(ImageRegion[0])
        # 將二進位資料轉換為 PIL 影像物件
        img_pil_image = Image.open(BytesIO(img_bytes_data))
        # 將 PIL 影像物件轉換為 NumPy 陣列
        back = np.array(img_pil_image)
        
        mask_bytes_data = base64.b64decode(MaskRegion[0])
        # 將二進位資料轉換為 PIL 影像物件
        mask_pil_image = Image.open(BytesIO(mask_bytes_data))
        # 將 PIL 影像物件轉換為 NumPy 陣列
        mask = np.array(mask_pil_image)

        if (len(mask.shape)== 3 ):
            mask = mask[:, :, 0]

        #這邊處理back跟註記合體
        back[:,:,0][mask!=0] = back[:,:,0][mask!=0]/2
        back[:,:,1][mask!=0] = np.clip((1.5*back[:,:,1][mask!=0].astype(np.int16)),0,255).astype(np.uint8)#back[:,:,1][mask!=0]/2
        back[:,:,2][mask!=0] = back[:,:,2][mask!=0]/2

        IntegrationRegion = nparray_to_pngbase64str(back)

        print(f'****************dcmimage.Get_Region.IntegrationRegion-HEAD{IntegrationRegion[:5]}**********************')
        print(f'****************dcmimage.Get_Region.IntegrationRegion-END{IntegrationRegion[-5:]}**********************')
        return (ImageRegion[0],MaskRegion[0],IntegrationRegion)

def prepare_send_ndarr_as_png(arr: np.ndarray):
    buf = BytesIO()
    format = 'png'
    arr.save(buf, format, quality=100)
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp

import subprocess
from pydicom.filereader import dcmread
import highdicom as hd
import openslide

#*****************************************************************************************************************************************
@image.route('/jellox/openslide/loadImage/', methods=['POST'])
def openslide_loadImage_api():
    print('****************loadImage**********************')
    collected = gc.collect()
    image_path = request.form.getlist('path')[0]
    mask_path = request.form.getlist('maskPath')[0]
    #level = int(request.form.getlist('level')[0])
    level = 0
    image_format = request.form.getlist('format')[0]
    maxPixel = request.form.getlist('maxPixel')[0]
    print(f'image_path : {image_path}')
    print(f'mask_path : {mask_path}')
    print(f'image_format : {image_format}')
    print(f'maxPixel : {maxPixel}')

    #要改switch
#########################
    if image_format.endswith((".dcm",".svs", ".ndpi", ".mrxs",".bif",".tif",".tiff")):
        if image_format.endswith(".dcm"):
            slide = DCM_Image()
        else:
            slide = OpenSlide_Image()
            
        
    elif image_format.endswith((".jpg", ".png")):
        slide = Normal_Image()

    (originW, originH) = slide.Load_Image(image_path)
    (ThumbnailW,ThumbnailH) = slide.Cal_Thumb_W_and_H(originW, originH, maxPixel)
    ThumbnailPNGImg = slide.Get_Thumbnail(image_path,ThumbnailW, ThumbnailH)

#########################
    #dcmimage = DCM_Image()

    #(originW, originH) = dcmimage.Load_Image(image_path)
    print(f'[result]originW : {originW}')
    print(f'[result]originH : {originH}')
    #(ThumbnailW,ThumbnailH) = dcmimage.Cal_Thumb_W_and_H(originW, originH, maxPixel)
    #ThumbnailPNGImg = dcmimage.Get_Thumbnail(image_path,ThumbnailW, ThumbnailH)

    response = {
        'originW': originW,
        'originH': originH,
        'ThumbnailW': ThumbnailW,
        'ThumbnailH': ThumbnailH,
        'ThumbnailPNGImg': ThumbnailPNGImg
    }

    return jsonify(response)

@image.route('/jellox/openslide/getRegion/', methods=['POST'])
def openslide_getRegion_api():
    print('****************getRegion**********************')
    collected = gc.collect()
    image_path = request.form.getlist('path')[0]
    mask_path = request.form.getlist('maskPath')[0]
    #level = int(request.form.getlist('level')[0])
    level = 0
    image_format = request.form.getlist('format')[0]
    x = int(float(request.form.getlist('x')[0]))
    y = int(float(request.form.getlist('y')[0]))
    w = int(float(request.form.getlist('w')[0]))
    h = int(float(request.form.getlist('h')[0]))

    print(f'image_path : {image_path}')
    print(f'mask_path : {mask_path}')
    print(f'image_format : {image_format}')    
    print(f'x : {x}')
    print(f'y : {y}')
    print(f'w : {w}')
    print(f'h : {h}')

    if image_format.endswith((".dcm",".svs", ".ndpi", ".mrxs",".bif",".tif",".tiff")):
        if image_format.endswith(".dcm"):
            slide = DCM_Image()
        else:
            slide = OpenSlide_Image()
            
        
    elif image_format.endswith((".jpg", ".png")):
        slide = Normal_Image()

    #dcmimage = DCM_Image()
    #ImageRegion,MaskRegion,IntegrationRegion = dcmimage.Get_Region(image_path, mask_path, x, y, w, h, request)
    ImageRegion,MaskRegion,IntegrationRegion = slide.Get_Region(image_path, mask_path, x, y, w, h, request)

    print(f'*****return getRegion*****')    
    response = {
        'ImageRegion': ImageRegion,
        'MaskRegion': MaskRegion,
        'IntegrationRegion': IntegrationRegion
    }

    return jsonify(response)

@image.route("/<path:path>", methods=['GET'])
def get_image(path):
    try:
        if not os.path.isfile(FILE_ROOT+path):
            return jsonify(
                status="ERROR",
                message="file not found"
            )
        slide = _get_slide(FILE_ROOT+path)
        format = "png"
        resp = make_response(slide.get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except:
        abort(404)

@image.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.png', methods=['GET'])
def tile(path, level, col, row):
    slide = _get_slide(FILE_ROOT+path)
    format = 'png'
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        print(f'[Slide] get tile error')
        abort(404)
    buf = BytesIO()
    tile.save(buf, format, quality=100)
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


@image.before_app_request
def _setup():
    from app import app
    opts = {
        'tile_size':254,
        'overlap':1,
        'limit_bounds':True,
    }
    app.cache = SlideCache(10, opts)

def _get_slide(path):
    from app import app
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        print('[Slide] preprocess error, ex: cache')
        abort(404)
