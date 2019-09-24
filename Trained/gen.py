import pyqrcode

def gen(file_name, url = 'photo.camcann.com'):
	big_code = pyqrcode.create(url, error='L', version=27, mode='binary')
	big_code.png(file_name, scale=6, module_color=[0, 0, 0, 128], background=[0xff, 0xff, 0xcc])
