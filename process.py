from tqdm import tqdm
import time


if __name__ == '__main__':

    # 假设你有100个文件要处理
    total_files = 100

    # 创建一个tqdm实例，设置总数和位置
    pbar_total = tqdm(total=total_files, position=0, leave=True)

    for i in range(total_files):
        # 对于每一个文件，你还有各种子任务要做
        sub_tasks = 50
        # 创建一个新的tqdm实例，设置总数和位置，注意位置应当在前一个的下一行
        pbar_sub = tqdm(total=sub_tasks, position=1, leave=False)

        for j in range(sub_tasks):
            # 处理子任务，并更新进度条
            time.sleep(0.01)  # 这里模拟了处理任务的延迟
            pbar_sub.update()

        # 完成所有子任务后，更新主进度条
        pbar_total.update()

        # 关闭子任务的进度条以避免混淆
        pbar_sub.close()

    # 处理完所有文件后，关闭主进度条
    pbar_total.close()