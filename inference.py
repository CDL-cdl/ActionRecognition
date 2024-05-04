import torch
import numpy as np
from network import C3D_model
import cv2

torch.backends.cudnn.benchmark = True

# 中心裁剪图像
def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)

# 简化的中心裁剪函数
def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    # 检测GPU是否可用，若不可用则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # 读取UCF类别标签
    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # 初始化C3D模型
    model = C3D_model.C3D(num_classes=101)
    torch.cuda.empty_cache()

    # 加载预训练权重
    checkpoint = torch.load('/mnt/iusers01/eee01/p52457dc/scratch/workspace/code/ActionRecognition/run/run_0/models/C3D-ucf101_epoch-99.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 将模型移动到GPU上并设置为评估模式
    model.to(device)
    model.eval()

    # 使用摄像头或者视频文件
    use_camera = True  # 如果为True，使用摄像头；如果为False，使用视频文件
    video_source = 0  # 0表示默认摄像头，也可以设置为视频文件路径

    if use_camera:
        cap = cv2.VideoCapture(video_source)
    else:
        video_file = 'path/to/your/video/file.mp4'
        cap = cv2.VideoCapture(video_file)

    retaining = True
    clip = []

    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue

        # 对每一帧进行中心裁剪和归一化处理
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        # 当clip中累积了足够的帧数（16帧），进行模型推理
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            with torch.no_grad():
                outputs = model.forward(inputs)

            # 使用Softmax获取类别概率分布
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            # 在视频中显示类别和概率信息
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 移除最旧的帧，保持clip中帧数为16
            clip.pop(0)

        # 显示视频帧
        cv2.imshow('result', frame)
        
        # 检测键盘按键事件，按下 'q' 键退出实时推理
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    # 释放摄像头和清除窗口
    torch.cuda.empty_cache()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
