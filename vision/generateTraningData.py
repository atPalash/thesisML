import rospy
from std_msgs.msg import String
import cv2
from imutils import paths
from objectSelection import objectSelectorOOP


def take_snap(data): #, camera_object, object_type):
    print 'data', data
    # snap_taken_publisher = rospy.Publisher('snap_taken', String, queue_size=0)
    # if data is 'ready':
    #     print 'take snap'
    #     camera_object.start_streaming()
    #     images = camera_object.detected_object_images
    #     # entire_image = camera_object.padded_image
    #     for img in images:
    #         image_rgb = img['RGB']
    #         image_depth = img['DEPTH']
    #         image_edged = img['EDGED']
    #         camera_object.write_data(object_type, image_rgb, image_edged, image_depth, image_pth='images/training_generator/')
    #         snap_taken_msg = 'ready'
    #         rospy.sleep(1)
    #         snap_taken_publisher.publish(snap_taken_msg)


if __name__ == "__main__":
    realsense_img_cols = 1280
    realsense_img_rows = 720
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding, True)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)

    object_list = {97: 'objA', 98: 'objB', 99: 'objC', 100: 'objD'}
    key_press = raw_input('Enter object name: ')

    if key_press not in object_list.values():
        print 'No object with name found'
        exit(0)

    rospy.init_node('camera_node_for_generating_training_data', anonymous=True)
    ready_for_snap_sub = rospy.Subscriber('ready_for_snap', String, take_snap)#, (camera, key_press))
    rospy.spin()



        # cv2.putText(image_rgb, objectIdentifier.predict(image_rgb),
        #             (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # count = 0
        # for c_pts in img['contour']:
        #     if count % 100 == 0:
        #         cv2.circle(entire_image, (int(c_pts[0][0][0]), int(c_pts[0][0][1])), 1, (255, 255, 0), -1)
        #         cv2.putText(entire_image, "({x}, {y}, {z})".format(x=int(c_pts[1][0] * 100), y=int(c_pts[1][1] * 100),
        #                 z=int(c_pts[1][2] * 100)), (int(c_pts[0][0][0]), int(c_pts[0][0][1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
        #     count = count + 1
        # cv2.imshow('detected_obj', image_rgb)
        # cv2.imshow('entire image', entire_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



