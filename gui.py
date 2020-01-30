import cv2
import detector
import tkinter
import tkinter.ttk
import tkinter.filedialog
import tkinter.messagebox

from PIL import Image
from PIL import ImageTk
import utils
import numpy as np

videoCapture = None

def loadCameraCallback():
    global videoCapture

    cache = utils.getSampleData()
    firstFrame = True
    if videoCapture is None:
        videoCapture = cv2.VideoCapture(0)

        while 1:
            ret, frame = videoCapture.read()

            if firstFrame:
                videoCaptureShape = frame.shape
                firstFrame = False

                canvas.config(height=videoCaptureShape[0], width=videoCaptureShape[1])

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processedFrame = detector.processFrame(frame, cache)
            pilImage = Image.fromarray(processedFrame)
            canvasImage = ImageTk.PhotoImage(pilImage)
            updateCanvasImage(canvasImage)

def loadImageCallback():
    try:
        filePath = tkinter.filedialog.askopenfilename()

        if not filePath:
            tkinter.messagebox.showwarning("Warn", "You must open an image")
            return

        openedImage = Image.open(filePath).convert("RGB")

        # Resize
        canvas.config(width=openedImage.size[0], height=openedImage.size[1])

        # convert the opened image from pil format to opencv format
        cache = utils.getSampleData()
        opencvImg = cv2.cvtColor(np.array(openedImage), cv2.COLOR_RGB2BGR)
        processed = detector.processFrame(opencvImg, cache)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # NOTE: The opened image is garbage collected right away and won't be displayed
        # Therefor, we're using the python's dynamic typing ability to create a new
        # field in the object to which the 'root' reference points to and store the
        # image reference there.
        processed = ImageTk.PhotoImage(Image.fromarray(processed))
        root.currentImage = processed
        canvas.create_image(0, 0, image=processed, anchor=tkinter.NW)
    except Exception as e:
        tkinter.messagebox.showerror("Error", "Couldn't open image: {}\nRaw error message: {}".format(filePath, e))

def updateCanvasImage(newImage):
    root.currentImage = newImage
    canvas.create_image(0, 0, image=newImage, anchor=tkinter.NW)
    root.update()

root = tkinter.Tk()
root.title("Currency detector")

canvas = tkinter.Canvas(root, bg="black")
canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)

loadImageButton = tkinter.ttk.Button(root, text="Load image", command=loadImageCallback)
loadImageButton.pack(anchor=tkinter.NW, side=tkinter.LEFT)

loadCameraButton = tkinter.ttk.Button(root, text="Start camera", command=loadCameraCallback)
loadCameraButton.pack(anchor=tkinter.N, side=tkinter.LEFT)
root.mainloop()
