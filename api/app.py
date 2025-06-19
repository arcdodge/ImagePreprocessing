from flask import Flask
import os
import config

# 匯入你在 image.py 裡定義的 Blueprint
from image import image as image_blueprint

# 讀取你的路徑設定
app = Flask(__name__)

# 把 Blueprint 掛到 /image 底下，這樣 image.py 裡定義的
# @image.route('/jellox/openslide/…') 就會對應到
# http://localhost:8000/image/jellox/openslide/…
app.register_blueprint(image_blueprint, url_prefix='/image')

app.config['FILE_ROOT'] = config.FILE_ROOT

if __name__ == '__main__':
    # host='0.0.0.0' 允許外部存取，本機測試可用 127.0.0.1
    app.run(host='0.0.0.0', port=64656, debug=True)
