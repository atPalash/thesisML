import cv2
from imutils import paths
from objectSelection import objectSelectorOOP
from objectSelection import objectIdentifier

if __name__ == "__main__":
    object_list = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
    selected_model = '/home/palash/thesis/thesisML/objectSelection/models/weights_best_RGB.hdf5'
    objectIdentifier = objectIdentifier.ObjectIdentfier(selected_model, 4, 3, object_list)

    realsense_img_cols = 1280
    realsense_img_rows = 720
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding, True)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    # imagePaths = sorted(list(paths.list_images('../objectSelection/images')))
    # for imagePath in imagePaths:

    camera.start_streaming()
    images = camera.detected_object_images
    entire_image = camera.padded_image
    for img in images:
        image_rgb = img['RGB']
        cv2.putText(image_rgb, objectIdentifier.predict(image_rgb),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        count = 0
        for c_pts in img['contour']:
            if count % 100 == 0:
                cv2.circle(entire_image, (int(c_pts[0][0][0]), int(c_pts[0][0][1])), 1, (255, 255, 0), -1)
                cv2.putText(entire_image, "({x}, {y}, {z})".format(x=int(c_pts[1][0] * 100), y=int(c_pts[1][1] * 100),
                        z=int(c_pts[1][2] * 100)), (int(c_pts[0][0][0]), int(c_pts[0][0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
            count = count + 1
        cv2.imshow('detected_obj', image_rgb)
        cv2.imshow('entire image', entire_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



