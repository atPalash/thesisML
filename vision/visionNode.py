import cv2

from objectSelection import objectSelectorOOP
from objectSelection import objectIdentifier

if __name__ == "__main__":
    object_list = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
    selected_model = '/home/palash/thesis/thesisML/objectSelection/models/weights_best_RGB.hdf5'
    objectIdentifier = objectIdentifier.ObjectIdentfier(selected_model, 4, 3, object_list)

    realsense_img_cols = 848
    realsense_img_rows = 480
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    camera.start_streaming()
    images = camera.detected_object_images
    for img in images:
        image_rgb = img['RGB']
        cv2.imshow('detected_obj', image_rgb)
        cv2.waitKey(0)
        print objectIdentifier.predict(image_rgb)



