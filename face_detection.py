"""
《人工智能基础》课程作业 - Assignment 1
实时摄像头人脸检测程序
作者：[你的姓名]
学号：[你的学号]
"""

import cv2
import os
import sys
import numpy as np
import time


def print_header():
    """打印程序标题"""
    print("=" * 60)
    print("《人工智能基础》课程作业 - 实时摄像头人脸检测")
    print("=" * 60)


def check_environment():
    """检查运行环境"""
    print("\n1. 检查运行环境...")

    try:
        import cv2
        cv_version = cv2.__version__
        print(f"   ✓ OpenCV 版本: {cv_version}")
    except ImportError:
        print("   ✗ 错误: OpenCV 未安装!")
        print("     请运行: pip install opencv-python")
        return False, None

    # 检查人脸检测模型
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("   ✗ 错误: 无法加载人脸检测模型!")
            return False, None
        print("   ✓ 人脸检测模型加载成功")
        return True, face_cascade
    except Exception as e:
        print(f"   ✗ 错误: 加载模型失败 - {e}")
        return False, None


def initialize_camera():
    """初始化摄像头"""
    print("\n2. 初始化摄像头...")

    # 尝试打开摄像头（0通常是默认摄像头）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("   ✗ 错误: 无法打开摄像头")
        print("     请检查:")
        print("     - 摄像头是否连接")
        print("     - 摄像头是否被其他程序占用")
        print("     - 尝试使用其他摄像头索引 (0, 1, 2...)")
        return None

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("   ✓ 摄像头初始化成功")
    return cap


def detect_faces_realtime(face_cascade, cap):
    """实时人脸检测"""
    print("\n3. 开始实时人脸检测")
    print("   操作说明:")
    print("   - 按 's' 键: 保存当前帧为图片")
    print("   - 按 'q' 键: 退出程序")
    print("   - 调整位置确保人脸在画面中")

    face_count_history = []  # 记录人脸数量历史
    save_count = 0  # 保存图片计数

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("   ✗ 错误: 无法读取摄像头画面")
            break

        # 创建副本用于显示
        display_frame = frame.copy()

        # 转换为灰度图进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 记录人脸数量
        face_count = len(faces)
        face_count_history.append(face_count)
        if len(face_count_history) > 10:  # 只保留最近10次记录
            face_count_history.pop(0)

        # 绘制人脸框和编号
        for i, (x, y, w, h) in enumerate(faces):
            # 绘制人脸矩形框 (绿色)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制人脸编号背景 (蓝色)
            cv2.rectangle(display_frame, (x, y - 25), (x + 40, y), (255, 0, 0), -1)

            # 绘制人脸编号文本 (白色)
            cv2.putText(display_frame, f'#{i + 1}', (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 添加状态信息
        avg_face_count = sum(face_count_history) / len(face_count_history)
        status_text = f"检测到人脸: {face_count} | 平均: {avg_face_count:.1f} | 按's'保存 | 按'q'退出"

        # 绘制状态栏
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示帧
        cv2.imshow('实时人脸检测 - 按 s 保存截图, 按 q 退出', display_frame)

        # 键盘输入处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            print("   退出程序")
            break
        elif key == ord('s'):  # 按 's' 保存截图
            save_count += 1
            filename = f"face_detection_capture_{save_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   ✓ 截图已保存: {filename}")

            # 如果有检测到人脸，也保存带框的版本
            if face_count > 0:
                marked_filename = f"face_detection_marked_{save_count}.jpg"
                cv2.imwrite(marked_filename, display_frame)
                print(f"   ✓ 带标记截图已保存: {marked_filename}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    return save_count


def main():
    """主函数"""
    print_header()

    # 检查环境
    env_ok, face_cascade = check_environment()
    if not env_ok:
        print("\n❌ 环境检查失败，请解决上述问题后重新运行程序")
        input("\n按 Enter 键退出...")
        return

    # 初始化摄像头
    cap = initialize_camera()
    if cap is None:
        input("\n按 Enter 键退出...")
        return

    # 执行实时人脸检测
    try:
        save_count = detect_faces_realtime(face_cascade, cap)

        print("\n" + "=" * 60)
        print("🎉 实时人脸检测程序执行完成！")
        print("=" * 60)

        if save_count > 0:
            print(f"\n✓ 共保存了 {save_count} 张截图")
            print("  这些图片可用于作业提交")
        else:
            print("\n⚠️  未保存任何截图")
            print("  运行程序时按 's' 键可以保存当前画面")

        print("\n作业提交说明:")
        print("1. 请使用保存的截图作为作业提交材料")
        print("2. 确保截图中包含人脸检测框")
        print("3. 将截图粘贴到作业文档中")

    except Exception as e:
        print(f"\n❌ 程序执行过程中发生错误: {e}")
        print("请检查摄像头和运行环境")

    input("\n按 Enter 键退出程序...")


if __name__ == "__main__":
    main()