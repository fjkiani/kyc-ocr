def check_imports():
    try:
        import cv2
        print("✅ OpenCV installed successfully")
    except ImportError as e:
        print("❌ OpenCV:", e)

    try:
        import keras_ocr
        print("✅ Keras-OCR installed successfully")
    except ImportError as e:
        print("❌ Keras-OCR:", e)

    try:
        import numpy
        print("✅ NumPy installed successfully")
    except ImportError as e:
        print("❌ NumPy:", e)

    try:
        import pytesseract
        print("✅ Pytesseract installed successfully")
    except ImportError as e:
        print("❌ Pytesseract:", e)

    try:
        import easyocr
        print("✅ EasyOCR installed successfully")
    except ImportError as e:
        print("❌ EasyOCR:", e)

    try:
        import matplotlib
        print("✅ Matplotlib installed successfully")
    except ImportError as e:
        print("❌ Matplotlib:", e)

if __name__ == "__main__":
    check_imports()