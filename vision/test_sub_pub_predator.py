#!/usr/bin/env python

import rospy
from std_msgs.msg import String

check_ = False

count = 0
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    global count
    count = count + 1
    if count is 10:
        global check_
        check_ = True

def talker():
    pub = rospy.Publisher('predator', String, queue_size=1)
    rospy.init_node('predator', anonymous=True)
    rospy.Subscriber('ready_for_snap', String, callback)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world from predator %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()
        global check_
        if check_:
            break

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
