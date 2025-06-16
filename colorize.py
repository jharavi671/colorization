import numpy as np
import cv2
from tkinter import Tk, filedialog, Button, Label, Toplevel, Frame, Scale, HORIZONTAL, simpledialog
from PIL import Image, ImageTk, ImageOps
import os
from tkinter import colorchooser

def compress_image(image, file_path, size_limit_kb):
    size_limit_bytes = size_limit_kb * 1024
    if file_path.endswith('.jpg'):
        for quality in range(100, 0, -5):
            temp_path = file_path + "_temp.jpg"
            image.save(temp_path, format='JPEG', quality=quality)
            if os.path.getsize(temp_path) <= size_limit_bytes:
                os.rename(temp_path, file_path)
                return
            os.remove(temp_path)
    elif file_path.endswith('.png'):
        image = image.convert('RGB')
        image.save(file_path, format='PNG', optimize=True)
        if os.path.getsize(file_path) > size_limit_bytes:
            messagebox.showinfo("Size Limit Exceeded", "Couldn't compress PNG below the limit.")
    elif file_path.endswith('.pdf'):
        image = image.convert('RGB')
        image.save(file_path, format='PDF')
        if os.path.getsize(file_path) > size_limit_bytes:
            messagebox.showinfo("Size Limit Exceeded", "Couldn't compress PDF below the limit.")


# Paths to load the model
DIR = r"D:\Colorizing-black-and-white-images-using-Python-master"
PROTOTXT = os.path.join(DIR, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model", "pts_in_hull.npy")
MODEL = os.path.join(DIR, "model", "colorization_release_v2.caffemodel")

# Load the Model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

image_path = None
colorized_image = None  # Store the colorized image globally
border_added = False
border_color = None  # Store the selected border color

def select_image():
    global image_path, colorized_image, border_color, border_added
    image_path = filedialog.askopenfilename()
    if image_path:  # Check if an image is selected
        image = cv2.imread(image_path)
        image = cv2.resize(image, (400, 400))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        original_label.config(image=image)
        original_label.image = image

        # Reset colorized image and border settings
        colorized_image = None
        border_color = None
        border_added = False

def colorize_image():
    global colorized_image
    if image_path is None:
        return  # If no image is selected, do nothing

    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")
    colorized = cv2.resize(colorized, (400, 400))
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    colorized = Image.fromarray(colorized)
    colorized_image = ImageTk.PhotoImage(colorized)

    # Display the colorized image in the right panel
    colorized_label.config(image=colorized_image)
    colorized_label.image = colorized_image

def save_colorized_image():
    global colorized_image
    if colorized_image is None:
        return  # If no colorized image is available, do nothing

    # Open a file dialog to choose the save location and format
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG files", "*.png"),
                                                          ("JPEG files", "*.jpg"),
                                                          ("PDF files", "*.pdf")])
    if file_path:
        # Convert the ImageTk PhotoImage back to PIL Image
        pil_image = ImageTk.getimage(colorized_image)
        
        # Save the image in the selected format
        if file_path.endswith('.jpg'):
            # Ask for compression quality
            quality = simpledialog.askinteger("Compression Quality", "Enter quality (1-100):", minvalue=1, maxvalue=100, initialvalue=85)
            if quality is not None:
                pil_image.save(file_path, format='JPEG', quality=quality)
        elif file_path.endswith('.png'):
            pil_image.save(file_path, format='PNG')
        elif file_path.endswith('.pdf'):
            # Ensure the image is in RGB mode before saving as PDF
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pil_image.save(file_path, format='PDF')

def edit_colorized_image():
    global colorized_image, border_added, border_color
    if colorized_image is not None:
        new_window = Toplevel()
        new_window.title("Edit Colorized Image")
        new_window.geometry("800x600")

        editable_image = ImageTk.getimage(colorized_image)
        undo_stack = [(editable_image, 0, 1.0)]
        redo_stack = []
        border_added = False
        border_color = None

        def save_state(image, brightness, contrast):
            undo_stack.append((image.copy(), brightness, contrast))
            redo_stack.clear()

        def update_display(image):
            img_tk = ImageTk.PhotoImage(image)
            colorized_label_in_new_window.config(image=img_tk)
            colorized_label_in_new_window.image = img_tk

        def add_border():
            global border_added, border_color
            if not border_added and undo_stack:
                pil_image, brightness, contrast = undo_stack[-1]
                border_color = colorchooser.askcolor(title="Choose Border Color")[1]

                if not border_color:  # If user cancels selection, do nothing
                    return

                bordered_image = ImageOps.expand(pil_image, border=20, fill=border_color)
                save_state(bordered_image, brightness, contrast)
                update_display(bordered_image)
                border_added = True

        def undo():
            if len(undo_stack) > 1:
                redo_stack.append(undo_stack.pop())
                image, brightness, contrast = undo_stack[-1]
                update_display(image)
                brightness_slider.set(brightness)
                contrast_slider.set(contrast)

        def redo():
            if redo_stack:
                restored_image, brightness, contrast = redo_stack.pop()
                undo_stack.append((restored_image, brightness, contrast))
                update_display(restored_image)
                brightness_slider.set(brightness)
                contrast_slider.set(contrast)

        def adjust_brightness_contrast(val=None, save=False):
            global border_added, border_color
            if undo_stack:
                brightness = brightness_slider.get()
                contrast = contrast_slider.get()
                
                last_image, _, _ = undo_stack[0]  # First saved image (original)

                # Extract image area without border
                border_size = 20 if border_added else 0
                img_cropped = last_image.crop((border_size, border_size, last_image.width - border_size, last_image.height - border_size)) if border_added else last_image

                img_array = np.array(img_cropped)

                # Apply brightness & contrast
                adjusted_image = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
                adjusted_image_pil = Image.fromarray(adjusted_image)

                # Reapply the border if added
                if border_added and border_color:
                    adjusted_image_pil = ImageOps.expand(adjusted_image_pil, border=20, fill=border_color)

                if save:
                    save_state(adjusted_image_pil, brightness, contrast)

                update_display(adjusted_image_pil)

        # Function to save the edited image
        def save_edited_image():
            if colorized_image is None:
                return  # If no colorized image is available, do nothing

            # Open a file dialog to choose the save location and format
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                       filetypes=[("PNG files", "*.png"),
                                                                  ("JPEG files", "*.jpg"),
                                                                  ("PDF files", "*.pdf")])
            if file_path:
                # Convert the ImageTk PhotoImage back to PIL Image
                edited_image = ImageTk.getimage(colorized_label_in_new_window.image)
                
                # Save the image in the selected format
                if file_path.endswith('.jpg'):
                    edited_image.save(file_path, format='JPEG')
                elif file_path.endswith('.png'):
                    edited_image.save(file_path, format='PNG')
                elif file_path.endswith('.pdf'):
                    # Ensure the image is in RGB mode before saving as PDF
                    if edited_image.mode != 'RGB':
                        edited_image = edited_image.convert('RGB')
                    edited_image.save(file_path, format='PDF')

        colorized_label_in_new_window = Label(new_window)
        colorized_label_in_new_window.pack(side="left", padx=10, pady=10)

        update_display(editable_image)

        left_frame = Frame(new_window)
        left_frame.pack(side="left", padx=10, pady=10)

        Button(left_frame, text="Add Border", command=add_border).pack(pady=10)
        Button(left_frame, text="Undo", command=undo).pack(pady=10)
        Button(left_frame, text="Redo", command=redo).pack(pady=10)

        brightness_slider = Scale(left_frame, from_=-100, to=100, orient=HORIZONTAL, label="Brightness",
                                  command=lambda val: adjust_brightness_contrast(val))
        brightness_slider.set(0)
        brightness_slider.pack(pady=10)

        contrast_slider = Scale(left_frame, from_=0.5, to=3.0, orient=HORIZONTAL, resolution=0.1, label="Contrast",
                                command=lambda val: adjust_brightness_contrast(val))
        contrast_slider.set(1.0)
        contrast_slider.pack(pady=10)

        Button(left_frame, text="Save Adjustments", command=lambda: adjust_brightness_contrast(None, save=True)).pack(pady=10)
        Button(left_frame, text="Save Edited Image", command=save_edited_image).pack(pady=10)

root = Tk()
root.title("Colorization GUI")

Button(root, text="Open Image", command=select_image).grid(row=0, column=0, padx=10, pady=10)
original_label = Label(root, borderwidth=2, relief="solid")
original_label.grid(row=1, column=0, padx=10, pady=10)

colorized_label = Label(root, borderwidth=2, relief="solid")
colorized_label.grid(row=1, column=1, padx=10, pady=10)

Button(root, text="Colorize Image", command=colorize_image).grid(row=0, column=1, padx=10, pady=10)
Button(root, text="Edit the Colorized Image", command=edit_colorized_image).grid(row=2, column=1, padx=10, pady=10)

# Add Save Button
Button(root, text="Save Colorized Image", command=save_colorized_image).grid(row=2, column=0, padx=10, pady=10)

root.mainloop()