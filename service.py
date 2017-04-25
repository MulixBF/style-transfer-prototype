import os

from flask import Flask, Response, request, send_file
from uuid import uuid1 as new_uuid
from PIL import Image, ImageOps
import random
import os.path
from prisma import Prisma
from face_detector import FaceDetector

app = Flask(__name__)

app.config['UPLOAD_DIR'] = 'uploads/'
app.config['CONVERT_DIR'] = 'converted/'

if not os.path.exists(app.config['UPLOAD_DIR']):
    os.makedirs(app.config['UPLOAD_DIR'])

if not os.path.exists(app.config['CONVERT_DIR']):
    os.makedirs(app.config['CONVERT_DIR'])


class HttpException(Exception):
    def __init__(self, code: int = 500, reason: str = None, exception: Exception = None):
        description = reason if reason is not None else str(exception) if exception is not None else None
        super().__init__(description if description is not None else 'Unspecified error')
        self.reason = description
        self.code = code
        self.exception = exception


prisma = Prisma()
face_detector = FaceDetector()


def _upload_image(request):

    assert request is not None

    if not request.files['image']:
        raise HttpException(code=400, reason='File not specified')

    image_data = request.files['image']

    filename = str(new_uuid()) + '.png'
    upload_path = os.path.join(app.config['UPLOAD_DIR'], filename)

    image = Image.open(image_data.stream)
    image.save(upload_path)

    return filename


def _convert_image(filename: str, converter: callable):

    original_path = os.path.join(app.config['UPLOAD_DIR'], filename)
    converted_path = os.path.join(app.config['CONVERT_DIR'], filename)

    image = Image.open(original_path)
    converted = converter(image)
    converted.save(converted_path)

    return converted_path


@app.route('/', methods=['POST'])
def convert():
    upload_path = _upload_image(request)
    converted_path = _convert_image(upload_path, process_image)
    return send_file(converted_path)


@app.errorhandler(HttpException)
def handle_http_exception(e):
    return Response(content_type='text/plain', response=e.reason, status=e.code)


def process_image(image):
    face = face_detector.get_face_image(image)
    style = random.choice(Prisma.STYLES)
    return prisma.process_image(face, style)


if __name__ == '__main__':
    app.run()
