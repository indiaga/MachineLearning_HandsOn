from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.lang import Builder
import requests
import json

class LoginApp(App):
    def build(self):
        self.title = "Login Screen"
        Window.size = (400,200)
        Window.clearcolor = (0.6,0.6,0.6,1)
        return Builder.load_file('CDSW_Image_KivyApp2.kv')
        # pass
    def login_button_action(self):
        url = 'https://reqres.in/api/login'
        #data = json.dumps({"email": "eve.holt@reqres.in","password": "cityslicka"})
        data = json.dumps({"email": self.root.ids.usernamevalue.text,"password": self.root.ids.passwordvalue.text})
        response = requests.post(url, data=data, headers={'Content-Type':'application/json'})

        print(response.text)
if __name__ == '__main__':
    LoginApp().run()