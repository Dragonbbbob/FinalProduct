from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import cv2
import supervision as sv
from gtts import gTTS
from typing import List
from inference.models.yolo_world.yolo_world import YOLOWorld
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Initialize the YOLO model
model = YOLOWorld(model_id="yolo_world/l")
classes = ["pencil", "eraser", "paper", "folder", "binder", "backpack", "book", "headphones"]
model.set_classes(classes)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Load the image
        image = cv2.imread(file_location)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read the image file.")

        # Run inference
        results = model.infer(image, confidence=0.05)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)

        # Create labels for detections
        labels = [
            f"{classes[class_id]} {confidence:0.3f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the image
        annotated_image = image.copy()
        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
        LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)

        # Save annotated image
        annotated_image_path = f"/tmp/annotated_{file.filename}"
        cv2.imwrite(annotated_image_path, annotated_image)

        # Convert detected objects to speech
        detected_objects = set(classes[class_id] for class_id in detections.class_id)
        if detected_objects:
            text = ", ".join(detected_objects) + "."
            tts = gTTS(text=text, lang='en')
            audio_file = f"/tmp/detected_objects.mp3"
            tts.save(audio_file)

        return {
            "message": "Inference completed successfully.",
            "detected_objects": list(detected_objects),
            "annotated_image_path": annotated_image_path,
            "audio_file_path": audio_file
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/image/")
async def download_image():
    return FileResponse("/tmp/annotated_image.jpg", media_type="image/jpeg")

@app.get("/download/audio/")
async def download_audio():
    return FileResponse("/tmp/detected_objects.mp3", media_type="audio/mpeg")

