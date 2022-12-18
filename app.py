from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
from main import model_prompt, processor_model, sequence_processor
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/donut")
def response():
    args = request.args.to_dict()

    if "question" in args.keys():
        task_ = args["task"]
        question_ = args["question"]
        model_, task_prompt_ = model_prompt(task_=task_, question_=question_)
    else:
        task_ = args["task"]
        model_, task_prompt_ = model_prompt(task_=task_)

    if "file" not in request.files:
        return "No file"
    else:
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        else:
            return "No selected file - extension must be png or jpg - Shape of the image could not be handled."
    image = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    print(f"image: {type(image)} size: {image.size}")
    outputs, processor = processor_model(model_, task_prompt_, image)
    response = sequence_processor(outputs, processor)
    print(f"response: \n {response}")
    return response


if __name__ == "__main__":
    app.debug = True
    app.run()
