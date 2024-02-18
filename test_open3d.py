import open3d as o3d
import numpy as np

# 创建一个OffScreenRenderer对象
renderer = o3d.visualization.rendering.OffscreenRenderer(1920,1080)
material = o3d.cuda.pybind.visualization.rendering.MaterialRecord()
material.base_color = np.array([1.0, 1.0, 0.0, 0.0])  # 注意这里有四个值，RGBA
# # 设置金属度为0.5
# material.metallic_img = np.array([[0.5]])

# # 设置粗糙度为0.2
# material.roughness_img = np.array([[0.2]])
# # 创建一些点的坐标数据
# points = np.array([[0.0, 0.0, 0.0],  # 点1的坐标
#                    [1.0, 0.0, 0.0],  # 点2的坐标
#                    [0.0, 1.0, 0.0],
#                    [100,100,100]]) # 点3的坐标

# # 创建一些点的颜色数据
# colors = np.array([[1.0, 0.0, 0.0],  # 点1的颜色（红色）
#                    [0.0, 1.0, 0.0],  # 点2的颜色（绿色）
#                    [0.0, 0.0, 1.0],
#                    [1,0,0]]) # 点3的颜色（蓝色）
num_points = 100
points = np.random.rand(num_points, 3)

# 生成随机颜色
colors = np.random.rand(num_points, 3)
# 创建一个点云对象并设置点的坐标数据
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
# 设置渲染参数
width = 1920  # 渲染结果宽度
height = 1080  # 渲染结果高度
# renderer.scene.scene.set_background([0, 0, 0, 0])  # 设置背景为透明
renderer.scene.scene.add_geometry("point_cloud", point_cloud,material)  # 添加点云到场景中

# fx = fy = 1000.0  # 焦距
# cx = 960.0       # 主点 x 坐标
# cy = 540.0       # 主点 y 坐标

# intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# extrinsic_matrix = np.eye(4)  # 假设单位矩阵
# intrinsic_width_px = 1920
# intrinsic_height_px = 1080

# renderer.setup_camera(intrinsic_matrix, extrinsic_matrix,intrinsic_width_px, intrinsic_height_px)

# 渲染并保存结果
image = renderer.render_to_image()
print(image.get_max_bound())
# print(image.PointCloud.points)
# image.save("result.png")
print(o3d.io.write_image("result.png", image, quality=9))
# 释放资源
# renderer.scene.scene.remove_geometry("point_cloud")
# renderer.clear_scene()