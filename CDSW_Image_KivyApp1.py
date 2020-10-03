from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.properties import ObjectProperty

import requests
import json

class MainApp(MDApp):

    property_parking_result= ObjectProperty(None)

    def build(self):
        self.title = "Car Parking Checker"
        Window.size = (600,500)
        Window.clearcolor = (0.6,0.6,0.6,1)
        screen = Builder.load_file('CDSW_Image_KivyApp1.kv')
        # 변수 포로퍼티 복사
        self.property_parking_result = screen.property_parking_result
        self.property_parking_result.font_name = 'malgun.ttf'

        return screen

    def image_button_action(self):
        print('image clicked-{0}')

    def parking_check_button_action(self, req_image):
        
        from PIL import Image
        from numpy import asarray
        from io import BytesIO
        import base64
        import matplotlib.pyplot as plt
        import requests
        import json

        print('request1-{0}'.format(req_image)) 
        source_image = req_image
        target_image = "images/car_parking.jpg"
        image = Image.open(source_image)
        orig_data = asarray(image)
        print('Original = {0}'.format(orig_data.shape))
           
        # 이미지 사이즈 조정 
        resize_image = image.resize((150, 150))
        resize_data = asarray(resize_image)
        print('Resized = {0}'.format(resize_data.shape))
        resize_image.save(target_image, "JPEG", quality=95 )
        # 이미지를 문자열로 변환
        with open(target_image, 'rb') as binary_file:
            binary_file_data = binary_file.read()
            base64_encoded_data = base64.b64encode(binary_file_data)
            image_to_text = base64_encoded_data.decode('utf-8')

        accKey = "m3s3i4ig6atyffm4pgib4ugaij7yfsqz"
        model_API_Key = "5a8ea1d9527eb761c2b394b81a988f1be5394225c5ba9c2c9955bb59ed3f72d1.94f1d4d570bca979e6d41856ce787ce272174e08d46253b1d4dfcd85f9fa554b"
        req_data = '{{"accessKey":"{0}","request":{{"image":"data:image/jpg;base64,{1}"}}}}'.format(accKey, image_to_text)
        req_head = 'Bearer {0}'.format(model_API_Key)

        r = requests.post('http://modelservice.hdp.cloudexchange.co.kr/model', data=req_data, 
            headers={'Content-Type': 'application/json', 'Authorization': req_head })

        print("http result-{0}".format(r.text))
        resdata = json.loads(r.text)

        if resdata["success"]:
            resval = resdata["response"]
            parking=json.loads(resval)
            resFree = parking["free"]
            resFull = parking["full"]
            if resFree > resFull:
                self.property_parking_result.text = "[size=18][color=#3368FF]주자장에 자리가 있습니다. ({0} 확률)[/color][/size]".format(resFree)
            else:
                self.property_parking_result.text = "[size=18][color=FF5733]주자장에 자리가 없습니다. ({0} 확률)[/color][/size]".format(resFull)
        else:
            print("Failed-{0}".format(r.text)) 

if __name__ == '__main__':
    MainApp().run()