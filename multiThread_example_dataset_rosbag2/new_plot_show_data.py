import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import shutil

from scipy.spatial.transform import Rotation as R
# bridge = CvBridge()

# 创建一个 VideoWriter 对象
# 参数依次为：输出文件名、编码方式、帧率、图像大小
output_video = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者使用其他编码方式如 'MJPG', 'MP4V', 'X264'
frame_rate = 30  # 可以根据需要调整帧率
img_size = (640, 480)  # 根据你的图像大小调整，例如 cv_img.shape[1], cv_img.shape[0]
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, img_size)
CAM_HZ=30
TRAIN_HZ=10
TASK_TIME=1000
CHECK_PIC_SAVE_FOLDER="WRC_juice2_bag_picture"

def check_folder():
    if not os.path.exists(CHECK_PIC_SAVE_FOLDER):
        os.makedirs(CHECK_PIC_SAVE_FOLDER)
    else:
        # 清空文件夹中的所有内容
        shutil.rmtree(CHECK_PIC_SAVE_FOLDER)
        os.makedirs(CHECK_PIC_SAVE_FOLDER)

    last_pic_folder = os.path.join(CHECK_PIC_SAVE_FOLDER, "last_pic")
    if not os.path.exists(last_pic_folder):
        os.makedirs(last_pic_folder)
    else:
        # 清空文件夹中的文件
        shutil.rmtree(last_pic_folder)
        os.makedirs(last_pic_folder)

def use_rosbag_to_show(bag_name):

    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    # 读取rosbag文件并提取所需数据
    bag = rosbag.Bag(bag_name, 'r')

    start_time = bag.get_start_time()
    end_time = start_time + TASK_TIME

    cmd_joint=[]
    cmd_joint_time_stamp=[]
    state_joint=[]
    state_joint_time_stamp=[]

    cmd_eef_pose=[]
    cmd_eef_pose_time_stamp=[]
    state_eef_pose=[]
    state_eef_pose_time_stamp=[]

    cmd_hand=[]
    cmd_hand_time_stamp=[]
    state_hand=[]
    state_hand_time_stamp=[]

    img=[]
    img_stamp=[]
    
    for topic, msg, t in bag.read_messages(topics=[ '/kuavo_arm_traj',\
                                                    '/robot_arm_q_v_tau',\
                                                    '/drake_ik/cmd_arm_hand_pose',\
                                                    '/drake_ik/real_arm_hand_pose', \
                                                    '/robot_hand_eff',\
                                                    '/robot_hand_position',\
                                                    '/head_camera/color/image_raw/compressed',\
                                                  ]):
        # msg_time = msg.header.stamp.to_sec()  # 将时间戳转换为秒
        # if msg_time > end_time:
        #     break  # 超过时间限制，停止读取
        
        if topic == '/kuavo_arm_traj':
            # cmd_joint.append(msg.position)
            cmd_joint_time_stamp.append(msg.header.stamp)
            cmd_joint.append(np.radians(msg.position)[:7])

        elif topic == '/robot_arm_q_v_tau':
            # 将弧度转换为角度
            # state_joint.append(np.rad2deg(msg.q))
            state_joint_time_stamp.append(msg.header.stamp)
            state_joint.append((msg.q)[:7])
            
        elif topic=='/drake_ik/cmd_arm_hand_pose':
            cmd_eef_pose_time_stamp.append(msg.header.stamp)
            xyz=np.array(msg.left_pose.pos_xyz)
            xyzw=np.array(msg.left_pose.quat_xyzw)
            rotation = R.from_quat(xyzw)
            # 转换为欧拉角 (默认是 'xyz' 顺序，单位是弧度)
            euler_angles = rotation.as_euler('xyz')
            xyzrpy=np.concatenate((xyz,euler_angles))
            cmd_eef_pose.append(xyzrpy)

        elif topic=='/drake_ik/real_arm_hand_pose':
            state_eef_pose_time_stamp.append(msg.header.stamp)
            xyz=np.array(msg.left_pose.pos_xyz)
            xyzw=np.array(msg.left_pose.quat_xyzw)
            rotation = R.from_quat(xyzw)
            # 转换为欧拉角 (默认是 'xyz' 顺序，单位是弧度)
            euler_angles = rotation.as_euler('xyz')
            xyzrpy=np.concatenate((xyz,euler_angles))
            state_eef_pose.append(xyzrpy)

        elif topic=='/robot_hand_eff':
            cmd_hand_time_stamp.append(msg.header.stamp)
            left_hand_pose=msg.data
            if left_hand_pose[-1]==0:
                grip=0
            elif left_hand_pose[-1]==90:
                grip=1
            else:
                print("hand pose error")
            cmd_hand.append(grip)

        elif topic=='/robot_hand_position':
            state_hand_time_stamp.append(msg.header.stamp)

            left_hand_pose=msg.left_hand_position
            if left_hand_pose[-1]==0:
                grip=0
            elif left_hand_pose[-1]==90:
                grip=1
            else:
                print("hand pose error")
            state_hand.append(grip)

        elif topic=='/head_camera/color/image_raw/compressed':
            img_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 使用 cv2 解码图像
            # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_img = cv2.resize(cv_img, (256, 256))
            # 将调整后的图像添加到 img 列表
            img.append(resized_img)
            # cv2.imshow("s",cv_img)
            # cv2.waitKey(1)

    # cmd_eef_pose_time_stamp=cmd_joint_time_stamp.copy()
    # state_eef_pose_time_stamp=state_joint_time_stamp.copy()

    bag.close()

    # 安全判断
    if len(cmd_joint) == 0 or len(state_joint) == 0:
        print("ROS bag file contains empty data for at least one topic.")
        return

    if len(cmd_joint) < 100 or len(state_joint) < 100:
        print("ROS bag file data count is too small (less than 100 data points). Please check again.")
        return
    

    aligned_state_joint = []
    aligned_cmd_joint = []
    aligned_state_hand=[]
    aligned_cmd_hand=[]
    aligned_cmd_eef_pose=[]
    aligned_state_eef_pose=[]
    
    drop=2
    img=img[drop:-drop]
    img_stamp=img_stamp[drop:-drop]
    assert len(img)==len(img_stamp)
    for stamp in img_stamp:
        stamp_sec=stamp.to_sec()
        idx_s = np.argmin(np.abs(np.array([t.to_sec() for t in state_joint_time_stamp]) - stamp_sec))
        aligned_state_joint.append(state_joint[idx_s])

        idx_a = np.argmin(np.abs(np.array([t.to_sec() for t in cmd_joint_time_stamp]) - stamp_sec))
        aligned_cmd_joint.append(cmd_joint[idx_a])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_hand_time_stamp]) - stamp_sec))
        aligned_state_hand.append(state_hand[idx_h])
        
        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_hand_time_stamp]) - stamp_sec))
        aligned_cmd_hand.append(cmd_hand[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_eef_pose_time_stamp]) - stamp_sec))
        aligned_cmd_eef_pose.append(cmd_eef_pose[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_eef_pose_time_stamp]) - stamp_sec))
        aligned_state_eef_pose.append(state_eef_pose[idx_h])
    

    aligned_cmd_joint = [list(item) for item in aligned_cmd_joint]
    aligned_state_joint = [list(item) for item in aligned_state_joint]
    aligned_cmd_eef_pose=[list(item) for item in aligned_cmd_eef_pose]
    aligned_state_eef_pose=[list(item) for item in aligned_state_eef_pose]

    print("all length==============>:\nimg_stamp,aligned_cmd_joint,aligned_state_joint,aligned_cmd_eef_pose,aligned_state_eef_pose,aligned_cmd_hand,aligned_state_hand")
    print(len(img_stamp),len(aligned_cmd_joint),len(aligned_state_joint),len(aligned_cmd_eef_pose),len(aligned_state_eef_pose),len(aligned_cmd_hand),len(aligned_state_hand))
    assert len(img_stamp)==len(aligned_cmd_joint)==len(aligned_state_joint)==len(aligned_cmd_eef_pose)==len(aligned_state_eef_pose)==len(aligned_cmd_hand)==len(aligned_state_hand)
    
    for i in range(len(img_stamp)):
        aligned_cmd_joint[i].append(aligned_cmd_hand[i])
        aligned_state_joint[i].append(aligned_state_hand[i])
        aligned_cmd_eef_pose[i].append(aligned_cmd_hand[i])
        aligned_state_eef_pose[i].append(aligned_state_hand[i])

 # s 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 # a 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    jump=CAM_HZ//TRAIN_HZ
    aligned_cmd_joint=np.array(aligned_cmd_joint)[::jump].astype(np.float32)
    aligned_state_joint=np.array(aligned_state_joint)[::jump].astype(np.float32)
    aligned_cmd_eef_pose=np.array(aligned_cmd_eef_pose)[::jump].astype(np.float32)
    aligned_state_eef_pose=np.array(aligned_state_eef_pose)[::jump].astype(np.float32)
    aligned_delta_cmd_eef_pose=None
    img=img[::jump]

    print("after jump, all length==============>:")
    print(len(img),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))
    assert len(img)==len(aligned_state_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)

    aligned_cmd_joint=aligned_cmd_joint[1:]
    aligned_state_joint=aligned_state_joint[1:]

    aligned_state_eef_pose=aligned_state_eef_pose[1:]
    aligned_delta_cmd_eef_pose=aligned_cmd_eef_pose[1:]-aligned_cmd_eef_pose[:-1]
    aligned_delta_cmd_eef_pose[:,6]=aligned_cmd_eef_pose[1:,6]
    aligned_cmd_eef_pose=aligned_cmd_eef_pose[1:]
    img=img[1:]
    
    print("after delete firet frame==============>:")
    print(len(img),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose))
    import matplotlib
    matplotlib.use('Agg')
    # 创建3行5列的图表并进行比较
    num_plots = min(len(aligned_cmd_eef_pose[0]), len(aligned_state_eef_pose[0]), 15)  # 限制最多只显示15个数据对比
    fig, axs = plt.subplots(3, 5, figsize=(16, 9))
    fig.suptitle(base_name, fontsize=16)
    for i in range(num_plots):
        kuavo_position = [data[i] for data in aligned_cmd_joint]
        # robot_q = [data[i] for data in aligned_state_joint]

        cmd_eef=[data[i] for data in aligned_cmd_eef_pose]
        state_eef=[data[i] for data in aligned_state_eef_pose]
        cmd_eef_delta=[data[i] for data in aligned_delta_cmd_eef_pose]
        row = i // 5
        col = i % 5
        # axs[row, col].plot(kuavo_position, label='/kuavo_arm_traj')
        # axs[row, col].plot(robot_q, label='/robot_arm_q_v_tau')
        axs[row, col].plot(cmd_eef, label='/cmd_eef')
        axs[row, col].plot(state_eef, label='/state_eef')
        axs[row, col].plot(cmd_eef_delta, label='/cmd_eef_delta')
        axs[row, col].set_title(f"motor {i+1} state")
        axs[row, col].legend()

    exampl_index=50
    print(f"example index {exampl_index}:")
    print(" cmd_joint:",aligned_cmd_joint[exampl_index],
          "\n state_joint:",aligned_state_joint[exampl_index],
          "\n aligned_delta_cmd_eef_pose:",aligned_delta_cmd_eef_pose[exampl_index],
          "\n state_eef:",aligned_state_eef_pose[exampl_index],
          "\n img shape:",img[exampl_index].shape)   

    plt.tight_layout()

    # 保存图片
    save_path = f"./WRC_juice2_bag_picture/{base_name}.png"
    plt.savefig(save_path)

    # 保存最后一张img
    cv2.imwrite(f"./{CHECK_PIC_SAVE_FOLDER}/last_pic/last_img.png",img[-1])
    # # 显示图片
    # plt.show()
    assert len(img)==len(aligned_state_eef_pose)==len(aligned_delta_cmd_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)
    print("all length==============>:img,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint")
    print(len(img),len(aligned_state_eef_pose),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))   
    return img,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint

if __name__ == "__main__":
    # use_rosbag_to_show('./rgb_60hz_bag_file/2024-04-26-17-59-42.bag')
    check_folder()
    import glob
    bagpath=glob.glob("/home/octo/hx/dataset/raw/rosbag_WRC_juice/pick_up_something*.bag")
    bagpath=sorted(bagpath)[30:35]
    print(len(bagpath))
    for path in bagpath:
        print("current path",path)
        use_rosbag_to_show(path)
