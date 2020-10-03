import requests

img_path = 'images/car_parking3.jpg' # Full
import base64

from PIL import Image
from numpy import asarray
from io import BytesIO

source_image = "images/car_parking2.jpg"
target_image = "images/car_parking22.jpg"
image = Image.open(source_image)
orig_data = asarray(image)
print('Original = {0}'.format(orig_data.shape))
# resize 할 이미지 사이즈 
resize_image = image.resize((150, 150))
resize_data = asarray(resize_image)
print('Resized = {0}'.format(resize_data.shape))
resize_image.save(target_image, "JPEG", quality=95 )

with open(target_image, 'rb') as binary_file:
    binary_file_data = binary_file.read()
    base64_encoded_data = base64.b64encode(binary_file_data)
    base64_message = base64_encoded_data
    #base64_message = base64_encoded_data.decode('utf-8')
print(base64_message)

#r = requests.post('http://modelservice.hdp.cloudexchange.co.kr/model', 
#    data='{"accessKey":"mw93jvkzneooz98vwwtusnt5298dpvl9","request":{"image":"data:image/png;base64,idjdhfhjhghggjhkghjfghhgdfgjfghgkghjjg"}}', 
#    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer <place model API key here>'})


ttt = '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'

im = Image.open(BytesIO(base64.b64decode(ttt)))
in_array = asarray(im)
print(in_array.shape)
im.show()



