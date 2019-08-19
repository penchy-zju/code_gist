import torch
import urllib3
from torchvision import models
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from flask import Flask, request, Response

app = Flask(__name__)

mask_rcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
mask_rcnn.to(device)
mask_rcnn.eval()

preprocess = transforms.Compose([transforms.ToTensor()])

http = urllib3.PoolManager(retries=2, timeout=10, num_pools=200, maxsize=200)


@app.route('/', methods=['POST'])
def generate():
    url = request.form.get('url', type=str, default=None)
    r = http.request('GET', url.strip())
    npa = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(npa, cv2.IMREAD_UNCHANGED)
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputImage = Image.fromarray(cv2_im)

    inputTensor = preprocess(inputImage)
    inputBatch = inputTensor.unsqueeze(0)
    inputBatch = inputBatch.to(device)

    with torch.no_grad():
        p = mask_rcnn(inputBatch)

    x1, y1, x2, y2 = p[0]['boxes'][0].cpu().detach().numpy()

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255))

    cv2.imwrite('box.jpg', img)

    image = cv2.imencode('.jpg', img)[1].tobytes()
    return Response(image, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20000, debug=False, threaded=True)
