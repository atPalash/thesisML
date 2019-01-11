from objectSelection import objectSelectorOOP

if __name__ == "__main__":
    realsense_img_cols = 848
    realsense_img_rows = 480
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    camera.start_streaming()

