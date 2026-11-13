from tkinter import *
from PIL import Image, ImageDraw
from predict import predict_digit

root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x500")

canvas = Canvas(root, width=300, height=300, bg="white")
canvas.pack(pady=20)

# Image used for prediction
image = Image.new("L", (300, 300), "white")
draw = ImageDraw.Draw(image)

result = Label(root, text="Predicted Digit: ", font=("Arial", 16))
result.pack()

def paint(event):
    x, y = event.x, event.y
    r = 8

    canvas.create_oval(x-r, y-r, x+r, y+r,
                       fill="black", outline="black")

    draw.ellipse((x-r, y-r, x+r, y+r),
                 fill="black", outline="black")

canvas.bind("<B1-Motion>", paint)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 300, 300), fill="white")
    result.config(text="Predicted Digit: ")

def predict():
    image.save("digit.png")
    digit = predict_digit("digit.png")
    result.config(text=f"Predicted Digit: {digit}")

Button(root, text="Predict",
       command=predict,
       width=15,
       bg="green",
       fg="white").pack(pady=5)

Button(root, text="Clear",
       command=clear_canvas,
       width=15,
       bg="red",
       fg="white").pack()

root.mainloop()