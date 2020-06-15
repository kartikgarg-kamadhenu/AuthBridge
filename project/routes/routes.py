from flask import request,jsonify,Blueprint
from project import app,create_logger,config
from werkzeug.utils import secure_filename
import os
import cv2
import base64
import imutils
from scipy import ndimage
import numpy as np
from project import ALLOWED_EXTENSIONS
from pyzbar import pyzbar
import math
from skimage.io import imread, imsave
from skimage import exposure
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity


dir=os.getcwd()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

i=['back_r\\000000.9.jpg','back_r\\000009.jpg','back_r\\000009_try.jpg','back_r\\000029.jpg']   #back
im_path=["000000.8.jpg","000001.8.jpg","000147.8.jpg","000313.8.jpg","000004.jpg"]              #front

           
def pattern(im,height,width):
    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width)
    cropped_bot = im[start_row:end_row , start_col:end_col]
    print("done")
    images=[cropped_bot]
    
    rKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    count=0
    for imagePath in images:
        image = imutils.resize(imagePath, height=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        b_t = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rKernel)
        
        gd_X = cv2.Sobel(b_t, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gd_X = np.absolute(gd_X)
        (minVal, maxVal) = (np.min(gd_X), np.max(gd_X))
        gd_X = (255 * ((gd_X - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gd_X, cv2.MORPH_CLOSE, rKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sKernel)
        thresh = cv2.erode(thresh, None, iterations=4)
        point = int(image.shape[1] * 0.05)
        thresh[:, 0:point] = 0
        thresh[:, image.shape[1] - point:] = 0
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(gray.shape[1])
            if ar > 3 and crWidth > 0.25:
                count=count+1
                break
            
            else:
                pass
        print(count)    
        return count,im

def notfound(im,countin):
       
    note=0
    if countin==0:
        img1 = ndimage.rotate(im, 90, reshape=True)
        h1,w1=img1.shape[:2]
        img2 = ndimage.rotate(im, -90, reshape=True)
        h2,w2=img2.shape[:2]
        img3 = ndimage.rotate(im, 180, reshape=True)
        h3,w3=img3.shape[:2]
        one,i1=pattern(img1,h1,w1)
        if one!=0:
            cv2.imwrite(os.path.join(dir,"new_img/0.jpg"),i1)
            
        else:
            two,i2=pattern(img2,h2,w2)
            if two!=0:
                cv2.imwrite(os.path.join(dir,"new_img/0.jpg"),i2)
                
            else:
                three,i3=pattern(img3,h3,w3)
                if three!=0:
                    cv2.imwrite(os.path.join(dir,"new_img/0.jpg"),i3)
                    
                else:
                    note=note+1
                    code="not readable"
                    print("not readable")
                    return code
    else:
        cv2.imwrite(os.path.join(dir,"new_img/0.jpg"),im)
    path=os.path.join(dir,"new_img/0.jpg")
    return path

def top_part(image,height, width):
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .5), int(width*.5)
    top1 = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(os.path.join(dir,"crop_f/c1.jpg"),top1)
    
    
    start_row, start_col = int(0), int(width*.5)
    end_row, end_col = int(height*.5), int(width)
    bot1 = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(os.path.join(dir,"crop_f/c2.jpg"),bot1)
    
    
    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width*.5)
    top2 = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(os.path.join(dir,"crop_f/c3.jpg"),top2)
    
    
    start_row, start_col = int(height * .5), int(width*.5)
    end_row, end_col = int(height), int(width)
    bot2 = image[start_row:end_row , start_col:end_col]
    cv2.imwrite(os.path.join(dir,"crop_f/c4.jpg"),bot2)
    return 0
   
def find_deg(file):
    image = cv2.imread(file)
    b_co = pyzbar.decode(image)
    l=0
    for b in b_co:
        (x, y, w, h) = b.rect
        Type = b.type
        l=len(Type)
        print("len: ",l)
    return l


def last_ang(file1):
    img_before = cv2.imread(file1)    
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    angles = []
    
    for x1, y1, x2, y2 in lines[0]:
        #cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
        
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img_before, median_angle)
    cv2.imwrite(os.path.join(dir,"crop_f/rot.jpg"),img_rotated)
    print( "Angle is {}".format(median_angle))
    
    return median_angle
    
    
def rot(image,height, width):
    
    paart=top_part(image,height, width)
    one=find_deg(os.path.join(dir,"crop_f/c1.jpg"))
    two=find_deg(os.path.join(dir,"crop_f/c2.jpg"))
    three=find_deg(os.path.join(dir,"crop_f/c3.jpg"))
    four=find_deg(os.path.join(dir,"crop_f/c4.jpg"))
    if two!=0:
        #print("Good Image")
        cv2.imwrite(os.path.join(dir,"crop_f/ori.jpg"),image)
        path=os.path.join(dir,"crop_f/ori.jpg")
    elif three !=0:
        #print("rot 180")
        img3 = ndimage.rotate(image, 180, reshape=True)
        cv2.imwrite(os.path.join(dir,"crop_f/ori.jpg"),img3)
        path=os.path.join(dir,"crop_f/ori.jpg")
    elif one!=0:
        #print("rot clock 90")
        img1 = ndimage.rotate(image, -90, reshape=True)
        cv2.imwrite(os.path.join(dir,"crop_f/ori.jpg"),img1)
        path=os.path.join(dir,"crop_f/ori.jpg")
        
    elif four!=0:
        #print("rot anti-clock 90")
        img4= ndimage.rotate(image, 90, reshape=True)
        cv2.imwrite(os.path.join(dir,"crop_f/ori.jpg"),img4)
        path=os.path.join(dir,"crop_f/ori.jpg")
    else:
        #print("save as itis")
        cv2.imwrite(os.path.join(dir,"crop_f/ori.jpg"),image)
        path=os.path.join(dir,"crop_f/ori.jpg")
    
    ang=last_ang(path)
    return ang


#########################################################################

def logarithmic_correction(file):
    img =imread(file)
    
    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)
    path=os.path.join(dir,'enhance/out.jpg')
    imsave(path,logarithmic_corrected)
    return path


def thresh_and_rescaling(path):
    img = imread(path)

    yen_threshold = threshold_yen(img)
    bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
    
    path=os.path.join(dir,'enhance/out1.jpg')
    
    imsave(path, bright)
    return path
    
def enhance_image(path):
    img = cv2.imread(path, -1)
    img=cv2.resize(img,(700,600))

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    path=os.path.join(dir,'enhance/out2.jpg')
    cv2.imwrite(path,result_norm)
    return path


# creating a blueprint
api=Blueprint('api',__name__)
@api.route('/upload', methods=["POST","GET"])
def upload():
    
    logger=create_logger(config)
    
    if request.method=="POST":
       
        try:
            
            #validates that the input is a json object
            data=request.get_json() 
            
            if 'image' not in data:
                return jsonify({"error":"no image part"})

            if 'type' not in data:
                return jsonify({"error":"no type part"})
            
            # checks if the input string is empty
            if data['image'] =="" or not(data['image'].strip()):
                logger.warning("no image string received") 
                return jsonify({"Message":"image string is empty"}),400
            
            image_bytes_string=data['image']
            
            # coverting string into bytes 
            image_str=image_bytes_string[2:-1].encode()
        
                          
            try:
                # to make the filenames unique
                string=image_str[11:21] 
                    
                # using this above string to create filenames
                filename=f"{string}.jpg"
                filename=secure_filename(filename)
                    
                # destination where image  will be saved
                file=os.path.join(app.config['UPLOAD_FOLDER'],filename)
                    
                # saving the image in uploads folder
                with open (file,'wb') as f:
                    f.write(base64.decodebytes(image_str))
                    
                            
                if data['type']=='front':
                            
                    try:
                        
                        image = cv2.imread(file)
                        
                        
                        #print(i[3])
                        height, width = image.shape[:2]
            
                        countin,im=pattern(image,height,width) 
                        notfound(image,countin)
                        
                        path=os.path.join(dir,"new_img/0.jpg")
                        
                        path1=logarithmic_correction(path)
                        path1=thresh_and_rescaling(path1)
                        path2=enhance_image(path1)
                        
                                
                        with open (path2,'rb') as f:
                            image_str=str(base64.encodebytes(f.read()))
                            
                        os.remove(file)
                        new_img=os.path.join(dir,'new_img')
                        enhance=os.path.join(dir,'enhance')
                        
                        
                        for file in os.listdir(new_img):
                            path=os.path.join(new_img,file)
                            os.remove(path)
                                
                        for file in os.listdir(enhance):
                            path=os.path.join(enhance,file)
                            os.remove(path)
                        
                        return jsonify({"image":image_str})
        
                    except:
            
                        logger.error("some internal error occured in function file or invalid image string")
                        return jsonify({"error":"something went wrong, please check the image string"}),400
            
                if data['type']=='back':
            
                    try:
                        image = cv2.imread(file)
                
                        #print(i[3])
                        height, width = image.shape[:2]

                        rot(image,height, width)
                        path=os.path.join(dir,"crop_f/ori.jpg")
                        path1=logarithmic_correction(path)
                        path1=thresh_and_rescaling(path1)
                        path2=enhance_image(path1)
                        
                                
                        with open (path2,'rb') as f:
                            image_str=str(base64.encodebytes(f.read()))
                            
                        os.remove(file)
                        crop_f=os.path.join(dir,'crop_f')
                        enhance=os.path.join(dir,'enhance')
                        
                        for file in os.listdir(crop_f):
                            path=os.path.join(crop_f,file)
                            os.remove(path)
                            
                        for file in os.listdir(enhance):
                            path=os.path.join(enhance,file)
                            os.remove(path)
                                
                        return jsonify({"image":image_str})
                        
                    except:
                
                        logger.error("some internal error occured in function file or invalid image string")
                        return jsonify({"error":"something went wrong, please check the image string"}),400
                
                logger.warning("type was incorrect")
                return jsonify({"error":"please check the type part"}),400
            
            except:
                        
                logger.warning(" some error in the image string input")
                return jsonify({"error":"please input the correct image string"}),400
        
        
        except:
        
            # when the input received is invalid
            logger.warning("invalid json input by the user")
            return jsonify({"error":"invalid json input"}),400
        
    else:
        
        logger.warning("GET request was made by the user")
        return jsonify({"error":"only POST requests are allowed"}),405



@api.route('/path', methods=["POST","GET"])
def path():
    
    logger=create_logger(config)
    
    if request.method=="POST":
       
        try:
            
            #validates that the input is a json object
            # to check if file is uploaded or not
            if 'file' not in request.files:
                return jsonify({"message":"no file part"}),400
        
            file=request.files['file']
            
            if file.filename=="":
                return jsonify({"message":"please upload a file"}),400
            
            type=request.form.get('type')
            
            # Checking if the fields are empty or not
            if type is None:
                return jsonify({"message":"no type part"}),400
        
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
        
                if type =='front':
                                    
                    try:
                                
                        image = cv2.imread(file_path)
                                
                                
                        #print(i[3])
                        height, width = image.shape[:2]
                    
                        countin,im=pattern(image,height,width) 
                        notfound(image,countin)
                                
                        path=os.path.join(dir,"new_img/0.jpg")
                                
                        path1=logarithmic_correction(path)
                        path1=thresh_and_rescaling(path1)
                        path2=enhance_image(path1)
                                
                                        
                        with open (path2,'rb') as f:
                            image_str=str(base64.encodebytes(f.read()))
                                    
                        os.remove(file_path)
                        new_img=os.path.join(dir,'new_img')
                        enhance=os.path.join(dir,'enhance')
                                
                                
                        for file in os.listdir(new_img):
                            path=os.path.join(new_img,file)
                            os.remove(path)
                                        
                        for file in os.listdir(enhance):
                            path=os.path.join(enhance,file)
                            os.remove(path)
                                
                        return jsonify({"image":image_str})
            
                
                    except:
                    
                        logger.error("some internal error occured")
                        return jsonify({"error":"something went wrong"}),400
                
                if type== 'back' :
                
                    try:
                
                        image = cv2.imread(file_path)
                        
                        height, width = image.shape[:2]
        
                        rot(image,height, width)
                        path=os.path.join(dir,"crop_f/ori.jpg")
                        path1=logarithmic_correction(path)
                        path1=thresh_and_rescaling(path1)
                        path2=enhance_image(path1)
                                
                                        
                        with open (path2,'rb') as f:
                            image_str=str(base64.encodebytes(f.read()))
                                    
                        os.remove(file_path)
                        crop_f=os.path.join(dir,'crop_f')
                        enhance=os.path.join(dir,'enhance')
                                
                        for file in os.listdir(crop_f):
                            path=os.path.join(crop_f,file)
                            os.remove(path)
                                    
                        for file in os.listdir(enhance):
                            path=os.path.join(enhance,file)
                            os.remove(path)
                                        
                        return jsonify({"image":image_str})
                                
                    except:
                        
                        logger.error("some internal error occured")
                        return jsonify({"error":"something went wrong"}),400
                        
                logger.warning("type was incorrect")
                return jsonify({"error":"please check the type part"}),400
                
            else:
                
                logger.warning("invalid file")
                return jsonify({"error":"invalid file"})
        
        except:
        
            # when the input received is invalid
            logger.warning("invalid input by the user")
            return jsonify({"error":"invalid input"}),400
        
    else:
        
        logger.warning("GET request was made by the user")
        return jsonify({"error":"only POST requests are allowed"}),405
        
