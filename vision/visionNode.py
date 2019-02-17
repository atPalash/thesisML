import cv2
from imutils import paths
from objectSelection import objectSelectorOOP
from objectSelection import objectIdentifier
from reebGraph import ReebGraph

if __name__ == "__main__":
    object_list = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
    selected_model = '/home/palash/thesis/thesisML/objectSelection/models/weights_best_RGB_3class.hdf5'
    objectIdentifier = objectIdentifier.ObjectIdentfier(selected_model, 4, 3, object_list)
    reeb_graph = ReebGraph.ReebGraph(gripper_width=1000, realtime_cam=True)

    realsense_img_cols = 1280
    realsense_img_rows = 720
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding, True, 3, 30,
                                               10, area_threshold=2000)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    # imagePaths = sorted(list(paths.list_images('../objectSelection/images')))
    # for imagePath in imagePaths:

    camera.start_streaming()
    camera.get_image_data()
    images = camera.detected_object_images
    entire_image = camera.padded_image
    for img in images:
        image_rgb = img['RGB']
        image_contour_xyz = img['contour']
        cv2.putText(image_rgb, objectIdentifier.predict(image_rgb),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        reeb_graph.get_image_contour(entire_image, image_contour_xyz)
        gripping_points = reeb_graph.gripping_points
        orientation = reeb_graph.object_orientation
        count = 0

        # for c_pts in img['contour']:
        print "gripping xyz"
        for c_pts in gripping_points:
            x_cord_1 = 0
            y_cord_1 = 0
            z_cord_1 = 0
            x_cord_2 = 0
            y_cord_2 = 0
            z_cord_2 = 0
            check = False
            for contour_pt in image_contour_xyz:
                if contour_pt[0][0][0] == c_pts[0][0] and contour_pt[0][0][1] == c_pts[0][1]:
                    x_cord_1 = contour_pt[1][0]
                    y_cord_1 = contour_pt[1][1]
                    z_cord_1 = contour_pt[1][2]
                    check = True
                if contour_pt[0][0][0] == c_pts[1][0] and contour_pt[0][0][1] == c_pts[1][1]:
                    x_cord_2 = contour_pt[1][0]
                    y_cord_2 = contour_pt[1][1]
                    z_cord_2 = contour_pt[1][2]
                    check =True
            if check:
                cv2.circle(entire_image, c_pts[0], 1, (255, 255, 0), -1)
                cv2.circle(entire_image, c_pts[1], 1, (0, 255, 255), -1)
            # cv2.putText(entire_image, "({o:.1f})".format(o=orientation), (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
            # cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_1, y=y_cord_1, z=z_cord_1), c_pts[0],
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
            # cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_2, y=y_cord_2, z=z_cord_2), c_pts[1],
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

                print x_cord_1, y_cord_1, z_cord_1
        cv2.imshow('detected_obj', image_rgb)
        cv2.imshow('entire image', entire_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
